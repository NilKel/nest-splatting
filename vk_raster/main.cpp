// vk_raster — standalone Vulkan HARDWARE-rasterizer port of the CONIC baked renderer.
//
// Pipeline per frame (all on GPU, timestamped per stage):
//   1. preprocess.comp  : per-Gauss project + CONIC precompute + SV color + depth key
//   2. radix sort       : 4 x 8-bit LSD passes over (depth key, index) — per PRIMITIVE,
//                         not per (primitive x tile) as the tiled SW path needs
//   3. instanced draw   : 8-vertex octagon per sorted Gaussian, fragment shader = the
//                         CONIC inner-loop body, ROP blend = front-to-back "under"
// Offscreen RGBA32F target; PSNR vs GT for every test camera; FPS protocol mirrors
// benchmark_baked (warmup + timed frames cycling the test set, GPU timestamps).
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>

#define VK_CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "Vulkan error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); exit(1); } } while (0)

struct Cam { uint32_t W, H; float view[16], proj[16], campos[3], tanfx, tanfy; std::vector<uint8_t> gt; };
struct Buf { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkDeviceSize size = 0; void* map = nullptr; };

static VkInstance g_inst; static VkPhysicalDevice g_phys; static VkDevice g_dev;
static VkQueue g_queue; static uint32_t g_qfam; static VkCommandPool g_pool;
static VkPhysicalDeviceMemoryProperties g_mem;
static float g_tsPeriod = 1.0f;
static std::string g_spvDir;
static uint64_t g_fragInv = 0, g_prims = 0, g_vsInv = 0, g_surv = 0; static uint32_t g_sx[8];

// ------------------------------------------------------------------ helpers
static uint32_t findMem(uint32_t bits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < g_mem.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (g_mem.memoryTypes[i].propertyFlags & props) == props) return i;
    fprintf(stderr, "no memory type\n"); exit(1);
}
static Buf createBuf(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    Buf b; b.size = size;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size = size; ci.usage = usage; ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(g_dev, &ci, nullptr, &b.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev, b.buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = findMem(mr.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(g_dev, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(g_dev, b.buf, b.mem, 0));
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) VK_CHECK(vkMapMemory(g_dev, b.mem, 0, size, 0, &b.map));
    return b;
}
static VkCommandBuffer beginOneShot() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool = g_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(g_dev, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi)); return cb;
}
static void endOneShot(VkCommandBuffer cb) {
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE)); VK_CHECK(vkQueueWaitIdle(g_queue));
    vkFreeCommandBuffers(g_dev, g_pool, 1, &cb);
}
static Buf uploadDeviceBuf(const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
    Buf stg = createBuf(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(stg.map, data, size);
    Buf dst = createBuf(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkCommandBuffer cb = beginOneShot(); VkBufferCopy c{0, 0, size}; vkCmdCopyBuffer(cb, stg.buf, dst.buf, 1, &c); endOneShot(cb);
    vkDestroyBuffer(g_dev, stg.buf, nullptr); vkFreeMemory(g_dev, stg.mem, nullptr);
    return dst;
}
static std::vector<char> readFile(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate); if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    size_t n = (size_t)f.tellg(); std::vector<char> v(n); f.seekg(0); f.read(v.data(), n); return v;
}
template <class T> static std::vector<T> readArr(const std::string& p) {
    auto v = readFile(p); std::vector<T> out(v.size() / sizeof(T)); memcpy(out.data(), v.data(), out.size() * sizeof(T)); return out;
}
static VkShaderModule loadSpv(const std::string& name) {
    auto code = readFile(g_spvDir + "/" + name + ".spv");
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; ci.codeSize = code.size(); ci.pCode = (const uint32_t*)code.data();
    VkShaderModule m; VK_CHECK(vkCreateShaderModule(g_dev, &ci, nullptr, &m)); return m;
}
static std::map<std::string, std::string> readMeta(const std::string& p) {
    std::map<std::string, std::string> m; std::ifstream f(p); std::string line;
    while (std::getline(f, line)) { auto e = line.find('='); if (e != std::string::npos) m[line.substr(0, e)] = line.substr(e + 1); }
    return m;
}
static VkDescriptorSetLayout makeLayout(const std::vector<VkDescriptorSetLayoutBinding>& b) {
    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; ci.bindingCount = (uint32_t)b.size(); ci.pBindings = b.data();
    VkDescriptorSetLayout l; VK_CHECK(vkCreateDescriptorSetLayout(g_dev, &ci, nullptr, &l)); return l;
}
static VkDescriptorSetLayoutBinding B(uint32_t i, VkDescriptorType t, VkShaderStageFlags s) { VkDescriptorSetLayoutBinding b{}; b.binding = i; b.descriptorType = t; b.descriptorCount = 1; b.stageFlags = s; return b; }
static VkPipeline makeCompute(VkShaderModule m, VkPipelineLayout pl) {
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
    ci.layout = pl; VkPipeline p; VK_CHECK(vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p)); return p;
}
static void writeBufDesc(VkDescriptorSet set, uint32_t binding, VkDescriptorType t, const Buf& b) {
    VkDescriptorBufferInfo bi{b.buf, 0, t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ? (VkDeviceSize)256 : VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w.dstSet = set; w.dstBinding = binding; w.descriptorCount = 1; w.descriptorType = t; w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(g_dev, 1, &w, 0, nullptr);
}

// ------------------------------------------------------------------ main
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: vk_raster <bundle_dir> [--warmup N] [--bench N] [--dump first.ppm]\n"); return 1; }
    std::string bundle = argv[1]; int warmup = 300, bench = 400; std::string dump; bool fp16 = false, byid = false, stats = false, percam = false; int elems = 4; std::string dumpprep, dumpall; float pad = 1.0f; int lpmode = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench")) bench = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump")) dump = argv[++i];
        else if (!strcmp(argv[i], "--fp16")) fp16 = true;
        else if (!strcmp(argv[i], "--percam")) percam = true;
        else if (!strcmp(argv[i], "--elems")) elems = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dumpprep")) dumpprep = argv[++i];
        else if (!strcmp(argv[i], "--dumpall")) dumpall = argv[++i];
        else if (!strcmp(argv[i], "--byid")) byid = true;
        else if (!strcmp(argv[i], "--stats")) stats = true;
        else if (!strcmp(argv[i], "--pad")) pad = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--lp")) lpmode = atoi(argv[++i]);
    }
    { std::string exe = argv[0]; auto s = exe.find_last_of('/'); g_spvDir = (s == std::string::npos ? std::string(".") : exe.substr(0, s)) + "/spv"; }

    // ---- bundle ----
    auto meta = readMeta(bundle + "/meta.txt");
    const uint32_t N = std::stoul(meta["N"]), K = std::stoul(meta["K"]);
    const int kernel_type = std::stoi(meta["kernel_type"]);
    const uint32_t Wp = std::stoul(meta["atlas_w"]), Hp = std::stoul(meta["atlas_h"]);
    const uint32_t nLayers = std::stoul(meta["n_layers"]), layerH = std::stoul(meta["layer_h"]);
    std::vector<uint32_t> cuts; { std::stringstream ss(meta["cuts"]); std::string t; while (std::getline(ss, t, ',')) cuts.push_back(std::stoul(t)); }
    const float atlas_scale = std::stof(meta["atlas_scale"]), atlas_offset = std::stof(meta["atlas_offset"]);
    const float sh_bias = std::stof(meta["sh_bias"]), res_bias = std::stof(meta["res_bias"]), compact_mult = std::stof(meta["compact_mult"]);
    auto means = readArr<float>(bundle + "/means.f32"), scales = readArr<float>(bundle + "/scales.f32"), rots = readArr<float>(bundle + "/rots.f32");
    auto opac = readArr<float>(bundle + "/opac.f32"), shapes = readArr<float>(bundle + "/shapes.f32");
    auto svs = readArr<float>(bundle + "/sv_sites.f32"), svt = readArr<float>(bundle + "/sv_tau.f32"), svc = readArr<float>(bundle + "/sv_colors.f32");
    auto aparams = readArr<float>(bundle + "/atlas_params.f32");
    auto bc7 = readFile(bundle + "/atlas.bc7");
    std::vector<Cam> cams;
    { auto cb = readFile(bundle + "/cams.bin"); const char* p = cb.data(); uint32_t n; memcpy(&n, p, 4); p += 4;
      for (uint32_t i = 0; i < n; i++) { Cam c; memcpy(&c.W, p, 4); p += 4; memcpy(&c.H, p, 4); p += 4;
        memcpy(c.view, p, 64); p += 64; memcpy(c.proj, p, 64); p += 64; memcpy(c.campos, p, 12); p += 12;
        memcpy(&c.tanfx, p, 4); p += 4; memcpy(&c.tanfy, p, 4); p += 4;
        size_t gsz = (size_t)3 * c.W * c.H; c.gt.assign(p, p + gsz); p += gsz; cams.push_back(std::move(c)); } }
    printf("[vk] N=%u K=%u kernel=%d atlas %ux%u -> %u layers x %u | %zu cams %ux%u\n", N, K, kernel_type, Wp, Hp, nLayers, layerH, cams.size(), cams[0].W, cams[0].H);

    // ---- instance / device ----
    { VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_3;
      VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ci.pApplicationInfo = &ai; VK_CHECK(vkCreateInstance(&ci, nullptr, &g_inst)); }
    { uint32_t n = 0; vkEnumeratePhysicalDevices(g_inst, &n, nullptr); std::vector<VkPhysicalDevice> d(n); vkEnumeratePhysicalDevices(g_inst, &n, d.data());
      g_phys = d[0]; for (auto& x : d) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(x, &p); if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { g_phys = x; g_tsPeriod = p.limits.timestampPeriod; printf("[vk] %s\n", p.deviceName); break; } } }
    vkGetPhysicalDeviceMemoryProperties(g_phys, &g_mem);
    { uint32_t n = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &n, nullptr); std::vector<VkQueueFamilyProperties> q(n); vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &n, q.data());
      g_qfam = ~0u; for (uint32_t i = 0; i < n; i++) if ((q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && q[i].timestampValidBits) { g_qfam = i; break; }
      if (g_qfam == ~0u) { fprintf(stderr, "no gfx+compute queue\n"); return 1; }
      float pr = 1.0f; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex = g_qfam; qi.queueCount = 1; qi.pQueuePriorities = &pr;
      VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES}; f13.dynamicRendering = VK_TRUE; f13.synchronization2 = VK_TRUE;
      VkPhysicalDeviceFeatures feat{}; feat.pipelineStatisticsQuery = VK_TRUE;
      VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.pNext = &f13; di.pEnabledFeatures = &feat; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
      VK_CHECK(vkCreateDevice(g_phys, &di, nullptr, &g_dev)); vkGetDeviceQueue(g_dev, g_qfam, 0, &g_queue);
      VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex = g_qfam; pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; VK_CHECK(vkCreateCommandPool(g_dev, &pi, nullptr, &g_pool)); }

    // ---- static buffers ----
    const VkBufferUsageFlags SSBO = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    Buf bMeans = uploadDeviceBuf(means.data(), means.size() * 4, SSBO), bScales = uploadDeviceBuf(scales.data(), scales.size() * 4, SSBO);
    Buf bRots = uploadDeviceBuf(rots.data(), rots.size() * 4, SSBO), bOpac = uploadDeviceBuf(opac.data(), opac.size() * 4, SSBO);
    Buf bShapes = uploadDeviceBuf(shapes.data(), shapes.size() * 4, SSBO), bSvS = uploadDeviceBuf(svs.data(), svs.size() * 4, SSBO);
    Buf bSvT = uploadDeviceBuf(svt.data(), svt.size() * 4, SSBO), bSvC = uploadDeviceBuf(svc.data(), svc.size() * 4, SSBO);
    Buf bAP = uploadDeviceBuf(aparams.data(), aparams.size() * 4, SSBO);
    Buf bPrep = createBuf((VkDeviceSize)N * 8 * 16, SSBO, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf bKeyA = createBuf((VkDeviceSize)N * 4, SSBO, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), bValA = createBuf((VkDeviceSize)N * 4, SSBO, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf bKeyB = createBuf((VkDeviceSize)N * 4, SSBO, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), bValB = createBuf((VkDeviceSize)N * 4, SSBO, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const uint32_t blockElems = 256u * (uint32_t)elems; const uint32_t numBlocks = (N + blockElems - 1) / blockElems;
    Buf bHist = createBuf((VkDeviceSize)256 * numBlocks * 4, SSBO, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf bArgs = createBuf(64, SSBO | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const uint32_t USLOTS = 4096; Buf bUni = createBuf((VkDeviceSize)256 * USLOTS, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // ---- BC7 atlas as a 2D array (layer cuts on shelf boundaries) ----
    VkImage atlasImg; VkDeviceMemory atlasMem; VkImageView atlasView; VkSampler atlasSamp;
    { VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_BC7_UNORM_BLOCK;
      ii.extent = {Wp, layerH, 1}; ii.mipLevels = 1; ii.arrayLayers = nLayers; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
      ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      VK_CHECK(vkCreateImage(g_dev, &ii, nullptr, &atlasImg));
      VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev, atlasImg, &mr);
      VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = findMem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      VK_CHECK(vkAllocateMemory(g_dev, &ai, nullptr, &atlasMem)); VK_CHECK(vkBindImageMemory(g_dev, atlasImg, atlasMem, 0));
      // staging: each layer = rows [cut_l, cut_{l+1}) of the source, 1 B/texel in BC7
      const size_t rowBlockBytes = (size_t)(Wp / 4) * 16;
      Buf stg = createBuf((VkDeviceSize)nLayers * layerH * Wp, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      memset(stg.map, 0, stg.size);
      std::vector<VkBufferImageCopy> regions;
      for (uint32_t l = 0; l < nLayers; l++) {
          uint32_t r0 = cuts[l], r1 = cuts[l + 1], span = r1 - r0;
          size_t srcOff = (size_t)(r0 / 4) * rowBlockBytes, len = (size_t)(span / 4) * rowBlockBytes, dstOff = (size_t)l * layerH * Wp;
          memcpy((char*)stg.map + dstOff, bc7.data() + srcOff, len);
          VkBufferImageCopy rc{}; rc.bufferOffset = dstOff; rc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, l, 1}; rc.imageExtent = {Wp, span, 1};
          regions.push_back(rc);
      }
      VkCommandBuffer cb = beginOneShot();
      VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; ib.image = atlasImg; ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, nLayers}; ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
      vkCmdCopyBufferToImage(cb, stg.buf, atlasImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)regions.size(), regions.data());
      ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
      endOneShot(cb); vkDestroyBuffer(g_dev, stg.buf, nullptr); vkFreeMemory(g_dev, stg.mem, nullptr);
      VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = atlasImg; vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY; vi.format = VK_FORMAT_BC7_UNORM_BLOCK; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, nLayers};
      VK_CHECK(vkCreateImageView(g_dev, &vi, nullptr, &atlasView));
      VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter = si.minFilter = VK_FILTER_LINEAR; si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod = 0.0f;
      VK_CHECK(vkCreateSampler(g_dev, &si, nullptr, &atlasSamp));
      printf("[vk] atlas uploaded: %.1f MB BC7 in %u layers\n", bc7.size() / 1048576.0, nLayers); }

    // ---- descriptor layouts / pipelines ----
    const auto SB = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, UB = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, CIS = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    const auto CS = VK_SHADER_STAGE_COMPUTE_BIT, VS = VK_SHADER_STAGE_VERTEX_BIT, FS = VK_SHADER_STAGE_FRAGMENT_BIT;
    std::vector<VkDescriptorSetLayoutBinding> lc = {B(0, UB, CS)}; for (uint32_t i = 1; i <= 12; i++) lc.push_back(B(i, SB, CS));
    VkDescriptorSetLayout layC = makeLayout(lc);
    VkDescriptorSetLayout layS = makeLayout({B(0, SB, CS), B(1, SB, CS), B(2, SB, CS), B(3, SB, CS), B(4, SB, CS), B(5, SB, CS)});
    VkDescriptorSetLayout layG = makeLayout({B(0, UB, VS | FS), B(1, SB, VS | FS), B(2, SB, VS), B(3, SB, VS | FS), B(4, CIS, FS), B(5, SB, FS)});
    VkPipelineLayout plC, plS, plG;
    { VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; ci.setLayoutCount = 1; ci.pSetLayouts = &layC; VK_CHECK(vkCreatePipelineLayout(g_dev, &ci, nullptr, &plC));
      VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4}; ci.pSetLayouts = &layS; ci.pushConstantRangeCount = 1; ci.pPushConstantRanges = &pcr; VK_CHECK(vkCreatePipelineLayout(g_dev, &ci, nullptr, &plS));
      ci.pushConstantRangeCount = 0; ci.pSetLayouts = &layG; VK_CHECK(vkCreatePipelineLayout(g_dev, &ci, nullptr, &plG)); }
    VkPipeline pPre = makeCompute(loadSpv("preprocess.comp"), plC);
    std::string es = "_e" + std::to_string(elems);
    VkPipeline pHist = makeCompute(loadSpv("radix_hist.comp" + es), plS), pScan = makeCompute(loadSpv("radix_scan.comp"), plS), pScat = makeCompute(loadSpv("radix_scatter.comp" + es), plS), pArgs = makeCompute(loadSpv("sort_args.comp"), plS);
    VkPipeline pGfx;
    { std::string sfx = std::string(byid ? "_id" : "") + (stats ? "_stats" : "");
      VkShaderModule vs = loadSpv("splat.vert" + std::string(byid ? "_id" : "")), fs = loadSpv("splat.frag" + sfx);
      VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VS, vs, "main", nullptr}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, FS, fs, "main", nullptr}};
      VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
      VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; vp.viewportCount = 1; vp.scissorCount = 1;
      VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
      VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      VkPipelineColorBlendAttachmentState ba{}; ba.blendEnable = VK_TRUE;
      ba.srcColorBlendFactor = VK_BLEND_FACTOR_DST_ALPHA; ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE; ba.colorBlendOp = VK_BLEND_OP_ADD;   // C += T * (alpha*feat)
      ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; ba.alphaBlendOp = VK_BLEND_OP_ADD;  // T *= (1-alpha)
      ba.colorWriteMask = 0xF;
      VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; cb.attachmentCount = 1; cb.pAttachments = &ba;
      VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}; VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; ds.dynamicStateCount = 2; ds.pDynamicStates = dyn;
      VkFormat fmt = fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT; VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO}; rci.colorAttachmentCount = 1; rci.pColorAttachmentFormats = &fmt;
      VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}; gi.pNext = &rci; gi.stageCount = 2; gi.pStages = st; gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia;
      gi.pViewportState = &vp; gi.pRasterizationState = &rs; gi.pMultisampleState = &ms; gi.pColorBlendState = &cb; gi.pDynamicState = &ds; gi.layout = plG;
      VK_CHECK(vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gi, nullptr, &pGfx)); }

    // ---- descriptor sets ----
    VkDescriptorPool dpool;
    { VkDescriptorPoolSize ps[3] = {{UB, 4}, {SB, 48}, {CIS, 2}}; VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; ci.maxSets = 8; ci.poolSizeCount = 3; ci.pPoolSizes = ps; VK_CHECK(vkCreateDescriptorPool(g_dev, &ci, nullptr, &dpool)); }
    auto allocSet = [&](VkDescriptorSetLayout l) { VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool = dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &l; VkDescriptorSet s; VK_CHECK(vkAllocateDescriptorSets(g_dev, &ai, &s)); return s; };
    VkDescriptorSet setC = allocSet(layC), setSA = allocSet(layS), setSB = allocSet(layS), setG = allocSet(layG);
    writeBufDesc(setC, 0, UB, bUni);
    { const Buf* cb[] = {&bMeans, &bScales, &bRots, &bOpac, &bShapes, &bSvS, &bSvT, &bSvC, &bPrep, &bKeyA, &bValA, &bArgs}; for (uint32_t i = 0; i < 12; i++) writeBufDesc(setC, i + 1, SB, *cb[i]); }
    { const Buf* a[] = {&bKeyA, &bValA, &bKeyB, &bValB, &bHist, &bArgs}; for (uint32_t i = 0; i < 6; i++) writeBufDesc(setSA, i, SB, *a[i]);
      const Buf* b[] = {&bKeyB, &bValB, &bKeyA, &bValA, &bHist, &bArgs}; for (uint32_t i = 0; i < 6; i++) writeBufDesc(setSB, i, SB, *b[i]); }
    writeBufDesc(setG, 0, UB, bUni); writeBufDesc(setG, 1, SB, bPrep); writeBufDesc(setG, 2, SB, bValA); writeBufDesc(setG, 3, SB, bAP);
    { VkDescriptorImageInfo ii{atlasSamp, atlasView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w.dstSet = setG; w.dstBinding = 4; w.descriptorCount = 1; w.descriptorType = CIS; w.pImageInfo = &ii; vkUpdateDescriptorSets(g_dev, 1, &w, 0, nullptr); }

    // ---- render target (all test cams share a resolution) ----
    const uint32_t W = cams[0].W, H = cams[0].H;
    for (auto& c : cams) if (c.W != W || c.H != H) { fprintf(stderr, "mixed camera resolutions not supported\n"); return 1; }
    VkImage rt; VkDeviceMemory rtMem; VkImageView rtView;
    { VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT; ii.extent = {W, H, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
      ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT; VK_CHECK(vkCreateImage(g_dev, &ii, nullptr, &rt));
      VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev, rt, &mr); VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = findMem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      VK_CHECK(vkAllocateMemory(g_dev, &ai, nullptr, &rtMem)); VK_CHECK(vkBindImageMemory(g_dev, rt, rtMem, 0));
      VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = rt; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; VK_CHECK(vkCreateImageView(g_dev, &vi, nullptr, &rtView)); }
    Buf bRead = createBuf((VkDeviceSize)W * H * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    Buf bStats = createBuf(32, SSBO | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    writeBufDesc(setG, 5, SB, bStats);
    VkQueryPool qps; { VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; ci.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS; ci.queryCount = 1;
      ci.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT | VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT;
      VK_CHECK(vkCreateQueryPool(g_dev, &ci, nullptr, &qps)); }
    VkQueryPool qp; { VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; ci.queryType = VK_QUERY_TYPE_TIMESTAMP; ci.queryCount = 4 * USLOTS; VK_CHECK(vkCreateQueryPool(g_dev, &ci, nullptr, &qp)); }
    VkCommandBuffer cmd; { VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool = g_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1; VK_CHECK(vkAllocateCommandBuffers(g_dev, &ai, &cmd)); }
    VkFence fence; { VkFenceCreateInfo ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VK_CHECK(vkCreateFence(g_dev, &ci, nullptr, &fence)); }

    // ---- per-frame ----
    auto setUniform = [&](const Cam& c, uint32_t slot) {
        float u[64] = {0};
        memcpy(u, c.view, 64); memcpy(u + 16, c.proj, 64); memcpy(u + 32, c.campos, 12);
        u[35] = (float)c.W; u[36] = (float)c.H; u[37] = c.tanfx; u[38] = c.tanfy; u[39] = (float)N; u[40] = (float)K; u[41] = (float)kernel_type;
        u[42] = sh_bias; u[43] = res_bias; u[44] = compact_mult; u[45] = 1.0f /*opacity_aware_beta*/; u[46] = 1.0f /*beta_mult*/; u[47] = 0.0f /*drop_lowpass*/; u[48] = 1.0f /*scale_modifier*/;
        u[49] = atlas_scale; u[50] = atlas_offset; u[51] = (float)Wp; u[52] = (float)layerH; u[53] = (float)nLayers; u[54] = pad; u[55] = (float)lpmode;
        memcpy((char*)bUni.map + (size_t)slot * 256, u, sizeof(u));
    };
    uint64_t ts[4]; double acc[4] = {0, 0, 0, 0};
    auto record = [&](const Cam& c, uint32_t slot, bool readback, bool first) {
        setUniform(c, slot);
        const uint32_t uoff = slot * 256, q0 = slot * 4;
        if (!first) {   // previous frame's draw read prep/keys; the next preprocess/sort rewrite them
            VkMemoryBarrier rb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; rb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT; rb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &rb, 0, nullptr, 0, nullptr);
        }
        vkCmdResetQueryPool(cmd, qp, q0, 4); vkCmdResetQueryPool(cmd, qps, 0, 1);
        vkCmdFillBuffer(cmd, bArgs.buf, 0, 64, 0);
        { VkMemoryBarrier fb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; fb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; fb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fb, 0, nullptr, 0, nullptr); }
        if (stats) { vkCmdFillBuffer(cmd, bStats.buf, 0, 32, 0); VkMemoryBarrier fb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; fb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; fb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT; vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &fb, 0, nullptr, 0, nullptr); }
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, q0 + 0);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        auto cbar = [&]() { vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr); };
        // 1. preprocess
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pPre); vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plC, 0, 1, &setC, 1, &uoff);
        vkCmdDispatch(cmd, (N + 255) / 256, 1, 1); cbar();
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, q0 + 1);
        // 1b. sort/draw args from the visible count (indirect everything downstream)
        { uint32_t be = blockElems; vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plS, 0, 1, &setSA, 0, nullptr);
          vkCmdPushConstants(cmd, plS, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &be);
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pArgs); vkCmdDispatch(cmd, 1, 1, 1);
          VkMemoryBarrier ab{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; ab.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; ab.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
          vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &ab, 0, nullptr, 0, nullptr); }
        // 2. radix sort over the M visible keys, 4 passes A->B->A->B->A
        for (uint32_t p = 0; p < 4; p++) {
            uint32_t shift = p * 8;
            VkDescriptorSet s = (p & 1) ? setSB : setSA;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plS, 0, 1, &s, 0, nullptr);
            vkCmdPushConstants(cmd, plS, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &shift);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pHist); vkCmdDispatchIndirect(cmd, bArgs.buf, 16); cbar();
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pScan); vkCmdDispatch(cmd, 1, 1, 1); cbar();
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pScat); vkCmdDispatchIndirect(cmd, bArgs.buf, 16); cbar();
        }
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, q0 + 2);
        // 3. draw
        { VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; ib.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
          ib.image = rt; ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; ib.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
          VkMemoryBarrier mb2{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
          vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 1, &mb2, 0, nullptr, 1, &ib); }
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}; ca.imageView = rtView; ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ca.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};   // C = 0, T = 1
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO}; ri.renderArea = {{0, 0}, {W, H}}; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{0, 0, (float)W, (float)H, 0, 1}; VkRect2D sc{{0, 0}, {W, H}}; vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pGfx); vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, plG, 0, 1, &setG, 1, &uoff);
        vkCmdBeginQuery(cmd, qps, 0, 0);
        vkCmdDrawIndirect(cmd, bArgs.buf, 32, 1, 0);
        vkCmdEndQuery(cmd, qps, 0);
        vkCmdEndRendering(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, q0 + 3);
        if (readback) {
            VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; ib.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ib.image = rt; ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; ib.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
            VkBufferImageCopy rc{}; rc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; rc.imageExtent = {W, H, 1};
            vkCmdCopyImageToBuffer(cmd, rt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, bRead.buf, 1, &rc);
            VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
        }
    };
    auto submitWait = [&]() {
        VK_CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        VK_CHECK(vkResetFences(g_dev, 1, &fence)); VK_CHECK(vkQueueSubmit(g_queue, 1, &si, fence)); VK_CHECK(vkWaitForFences(g_dev, 1, &fence, VK_TRUE, UINT64_MAX));
    };
    auto beginCmd = [&]() {
        VK_CHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    };
    auto readTs = [&](uint32_t slot, double* out) {
        VK_CHECK(vkGetQueryPoolResults(g_dev, qp, slot * 4, 4, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        out[0] = (ts[3] - ts[0]) * g_tsPeriod * 1e-6; out[1] = (ts[1] - ts[0]) * g_tsPeriod * 1e-6; out[2] = (ts[2] - ts[1]) * g_tsPeriod * 1e-6; out[3] = (ts[3] - ts[2]) * g_tsPeriod * 1e-6;
    };
    // single-frame path (quality pass, readback + stats)
    auto frame = [&](const Cam& c, bool readback, double* out) {
        beginCmd(); record(c, 0, readback, true); submitWait();
        if (stats) { uint64_t st[3]; VK_CHECK(vkGetQueryPoolResults(g_dev, qps, 0, 1, sizeof(st), st, sizeof(st), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            const uint32_t* sv_ = (const uint32_t*)bStats.map; g_vsInv = st[0]; g_prims = st[1]; g_fragInv = st[2]; g_surv = sv_[0]; memcpy(g_sx, sv_, 32); }
        if (out) readTs(0, out);
    };

    // ---- quality: every test camera once ----
    double psnr_sum = 0;
    for (size_t ci = 0; ci < cams.size(); ci++) {
        frame(cams[ci], true, nullptr);
        const uint8_t* gt = cams[ci].gt.data();
        std::vector<float> imgv; const float* img;
        if (fp16) {   // decode RGBA16F readback
            const uint16_t* hp = (const uint16_t*)bRead.map; imgv.resize((size_t)W * H * 4);
            for (size_t i = 0; i < imgv.size(); i++) { uint16_t h = hp[i]; uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float v;
                if (e == 0) v = ldexpf((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = ldexpf((float)(m | 1024), (int)e - 25);
                imgv[i] = s ? -v : v; }
            img = imgv.data();
        } else img = (const float*)bRead.map;
        double mse[3] = {0, 0, 0}; const size_t npx = (size_t)W * H;
        for (size_t i = 0; i < npx; i++) for (int ch = 0; ch < 3; ch++) { double d = (double)img[i * 4 + ch] - gt[(size_t)ch * npx + i] / 255.0; mse[ch] += d * d; }
        double ps = 0; for (int ch = 0; ch < 3; ch++) ps += 20.0 * log10(1.0 / sqrt(mse[ch] / npx)); ps /= 3.0;   // benchmark_baked: per-channel PSNR, mean
        psnr_sum += ps;
        if (!dumpall.empty()) {   // raw fp32 RGB, row-major, for scripts/eval_vk_renders.py
            char nm[512]; snprintf(nm, sizeof(nm), "%s/%03zu.f32", dumpall.c_str(), ci); FILE* f = fopen(nm, "wb");
            std::vector<float> rgb((size_t)W * H * 3); for (size_t i = 0; i < npx; i++) for (int ch = 0; ch < 3; ch++) rgb[i * 3 + ch] = img[i * 4 + ch];
            fwrite(rgb.data(), 4, rgb.size(), f); fclose(f);
        }
        if (ci == 0 && !dumpprep.empty()) {
            Buf hb = createBuf(bPrep.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            VkCommandBuffer cb = beginOneShot(); VkBufferCopy c{0, 0, bPrep.size}; vkCmdCopyBuffer(cb, bPrep.buf, hb.buf, 1, &c); endOneShot(cb);
            FILE* f = fopen(dumpprep.c_str(), "wb"); fwrite(hb.map, 1, bPrep.size, f); fclose(f); printf("[vk] prep dumped (%zu B)\n", (size_t)bPrep.size);
        }
        if (ci == 0 && !dump.empty()) {
            FILE* f = fopen(dump.c_str(), "wb"); fprintf(f, "P6\n%u %u\n255\n", W, H);
            for (size_t i = 0; i < npx; i++) for (int ch = 0; ch < 3; ch++) { float v = img[i * 4 + ch]; v = v < 0 ? 0 : (v > 1 ? 1 : v); fputc((int)(v * 255.0f + 0.5f), f); }
            fclose(f);
        }
    }
    printf("[vk] PSNR (mean over %zu test cams): %.3f dB\n", cams.size(), psnr_sum / cams.size());
    if (stats) { printf("[vk] STATS (last cam): frag invocations %llu (incl. helpers) | non-helper %u (%.1f%%) | survivors %u (%.1f%% of non-helper)\n",
                      (unsigned long long)g_fragInv, g_sx[1], 100.0 * g_sx[1] / std::max<uint64_t>(1, g_fragInv), g_sx[0], 100.0 * g_sx[0] / std::max<uint32_t>(1, g_sx[1]));
                 printf("[vk]   culls of non-helper: denom<0.1 %u (%.1f%%) | rho3d>=9 %u (%.1f%%) | alpha<1/255 %u (%.1f%%) | power>0 %u\n",
                      g_sx[2], 100.0 * g_sx[2] / std::max<uint32_t>(1, g_sx[1]), g_sx[3], 100.0 * g_sx[3] / std::max<uint32_t>(1, g_sx[1]), g_sx[4], 100.0 * g_sx[4] / std::max<uint32_t>(1, g_sx[1]), g_sx[5]);
                 printf("[vk]   fragments from CIRCLE-FALLBACK surfels: %u (%.1f%% of all) of which rho3d>=9 culled: %u\n", g_sx[6], 100.0 * g_sx[6] / std::max<uint32_t>(1, g_sx[1]), g_sx[7]); }

    // ---- FPS: warmup + timed frames recorded back-to-back in ONE submission, cycling test
    //      cameras. Per-frame fence waits let the GPU idle and downclock between frames
    //      (measured: 15x slower preprocess on the first frames after a host sync); the CUDA
    //      bench enqueues its 400 renders without syncing, so we do the same.
    if (bench > 0) {
        if ((uint32_t)(warmup + bench) > USLOTS) { fprintf(stderr, "warmup+bench > %u\n", USLOTS); return 1; }
        beginCmd();
        for (int i = 0; i < warmup + bench; i++) record(cams[i % cams.size()], (uint32_t)i, false, i == 0);
        submitWait();
        std::vector<double> total;
        for (int i = 0; i < bench; i++) { double o[4]; readTs((uint32_t)(warmup + i), o); total.push_back(o[0]); for (int k = 0; k < 4; k++) acc[k] += o[k];
            if (percam && i < (int)cams.size()) printf("[vk]   cam %2d: %.3f ms (pre %.3f sort %.3f raster %.3f)\n", i, o[0], o[1], o[2], o[3]); }
        if (percam) { std::vector<double> srt = total; std::sort(srt.begin(), srt.end()); double med = srt[bench / 2]; int nslow = 0;
            for (int i = 0; i < bench; i++) { double o[4]; readTs((uint32_t)(warmup + i), o); if (o[0] > 3.0 * med) { if (nslow < 12) printf("[vk]   SLOW frame %3d (cam %2d): %.3f ms  pre %.3f  sort %.3f  raster %.3f\n", i, (int)(i % cams.size()), o[0], o[1], o[2], o[3]); nslow++; } }
            printf("[vk]   slow frames (>3x median %.3f ms): %d of %d\n", med, nslow, bench); }
        double mean_ms = acc[0] / bench; std::sort(total.begin(), total.end());
        printf("[vk] GPU frame: mean %.4f ms (median %.4f, p95 %.4f) -> %.1f FPS\n", mean_ms, total[bench / 2], total[(size_t)(bench * 0.95)], 1000.0 / mean_ms);
        printf("[vk]   preprocess %.4f ms | sort %.4f ms | raster %.4f ms   (%.0f%% / %.0f%% / %.0f%%)\n", acc[1] / bench, acc[2] / bench, acc[3] / bench,
               100 * acc[1] / acc[0], 100 * acc[2] / acc[0], 100 * acc[3] / acc[0]);
    }
    return 0;
}

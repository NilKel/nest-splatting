// pybind11 binding for richgel999/bc7enc — single-block + bulk image encoder.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "src/bc7enc.h"
#include <vector>
#include <thread>
#include <atomic>

namespace py = pybind11;

static bool g_inited = false;

static void ensure_init() {
    if (!g_inited) {
        bc7enc_compress_block_init();
        g_inited = true;
    }
}

// Encode an HxWx4 uint8 RGBA image as BC7. Returns a flat bytes buffer of
// 16 bytes per 4x4 block, in row-major block order (rows of blocks first).
// H and W are padded up to multiples of 4 internally.
py::bytes encode_image_rgba(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
    int uber_level = 1,
    bool perceptual = false,
    int n_threads = 0)
{
    ensure_init();
    if (img.ndim() != 3 || img.shape(2) != 4)
        throw std::runtime_error("encode_image_rgba: expected HxWx4 uint8 array");
    const int H = (int)img.shape(0);
    const int W = (int)img.shape(1);
    const int Hb = (H + 3) / 4;     // blocks per column
    const int Wb = (W + 3) / 4;     // blocks per row
    const int n_blocks = Hb * Wb;
    const uint8_t* src = img.data();

    // Output buffer (16 B/block).
    std::vector<uint8_t> out(n_blocks * 16, 0);

    // Encoder params (shared across all threads — read-only).
    bc7enc_compress_block_params params;
    bc7enc_compress_block_params_init(&params);
    if (perceptual)
        bc7enc_compress_block_params_init_perceptual_weights(&params);
    else
        bc7enc_compress_block_params_init_linear_weights(&params);
    params.m_uber_level = (uint32_t)std::min(std::max(uber_level, 0), (int)BC7ENC_MAX_UBER_LEVEL);
    params.m_max_partitions = 16;  // moderate quality/speed tradeoff
    params.m_mode17_partition_estimation_filterbank = true;

    if (n_threads <= 0)
        n_threads = std::max(1, (int)std::thread::hardware_concurrency());

    auto worker = [&](int b_start, int b_end) {
        color_rgba block_pix[16];
        for (int b = b_start; b < b_end; b++) {
            const int by = b / Wb;
            const int bx = b - by * Wb;
            const int y0 = by * 4;
            const int x0 = bx * 4;
            // Gather 4×4 pixels (clamp to image edge for padding).
            for (int dy = 0; dy < 4; dy++) {
                for (int dx = 0; dx < 4; dx++) {
                    int y = std::min(y0 + dy, H - 1);
                    int x = std::min(x0 + dx, W - 1);
                    const uint8_t* p = src + ((size_t)y * W + x) * 4;
                    block_pix[dy * 4 + dx].m_c[0] = p[0];
                    block_pix[dy * 4 + dx].m_c[1] = p[1];
                    block_pix[dy * 4 + dx].m_c[2] = p[2];
                    block_pix[dy * 4 + dx].m_c[3] = p[3];
                }
            }
            bc7enc_compress_block(out.data() + (size_t)b * 16, block_pix, &params);
        }
    };

    if (n_threads == 1) {
        worker(0, n_blocks);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(n_threads);
        const int chunk = (n_blocks + n_threads - 1) / n_threads;
        for (int t = 0; t < n_threads; t++) {
            int s = t * chunk;
            int e = std::min(s + chunk, n_blocks);
            if (s >= e) break;
            threads.emplace_back(worker, s, e);
        }
        for (auto& th : threads) th.join();
    }

    return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
}

PYBIND11_MODULE(bc7encoder, m) {
    m.doc() = "BC7 encoder (richgel999/bc7enc) python binding";
    m.def("encode_image_rgba", &encode_image_rgba,
          py::arg("img"), py::arg("uber_level") = 1, py::arg("perceptual") = false,
          py::arg("n_threads") = 0,
          "Encode an HxWx4 uint8 RGBA image to a flat BC7 byte stream. "
          "Returns 16 bytes per 4x4 block, row-major.");
}

#!/bin/bash
#=============================================================================
# Git Safety Verification Script
# Run this anytime to verify your repository is properly protected
#=============================================================================

echo "🔍 === NEST-SPLATTING GIT SAFETY CHECK ==="
echo ""

# Check if any forbidden files are tracked
echo "📋 Checking for tracked forbidden files..."
FORBIDDEN_FILES=$(git ls-files | grep -E "(data/|outputs/|metrics_reports/|checkpoints/|logs/|tmp/|cache/|wandb/|tensorboard/|runs/|slurm_jobs/|\.pth$|\.pt$|\.npz$|\.npy$|\.pkl$|\.cache$|\.h5$|\.hdf5$|\.mat$|\.bin$|test_metrics\.txt|train_metrics\.txt|point_cloud\.ply)" || true)

if [ ! -z "$FORBIDDEN_FILES" ]; then
    echo "⚠️  WARNING: The following FORBIDDEN files are being tracked:"
    echo "$FORBIDDEN_FILES"
    echo ""
    echo "🔧 Run this to fix:"
    echo "   git rm --cached <filename>"
    echo ""
    SAFETY_STATUS="COMPROMISED"
else
    echo "✅ All clear - no forbidden files are tracked"
    SAFETY_STATUS="PROTECTED"
fi

echo ""
echo "🛡️  === PROTECTION LAYER STATUS ==="

# Check .gitignore protection
if grep -q "outputs/" .gitignore && grep -q "data/" .gitignore && grep -q "metrics_reports/" .gitignore; then
    echo "✅ Root .gitignore protection: ACTIVE"
    GITIGNORE_STATUS="ACTIVE"
else
    echo "❌ Root .gitignore protection: MISSING"
    GITIGNORE_STATUS="MISSING"
fi

# Check directory-level protection
DIRECTORY_PROTECTION="ACTIVE"
for dir in "data" "outputs" "metrics_reports"; do
    if [ -f "$dir/.gitignore" ]; then
        echo "✅ $dir/.gitignore protection: ACTIVE"
    else
        echo "❌ $dir/.gitignore protection: MISSING"
        DIRECTORY_PROTECTION="MISSING"
    fi
done

# Check pre-commit hook protection
if [ -f .git/hooks/pre-commit ] && [ -x .git/hooks/pre-commit ]; then
    echo "✅ Pre-commit hook protection: ACTIVE"
    HOOK_STATUS="ACTIVE"
else
    echo "❌ Pre-commit hook protection: MISSING"
    HOOK_STATUS="MISSING"
fi

echo ""
echo "📊 === OVERALL PROTECTION STATUS ==="

if [ "$SAFETY_STATUS" = "PROTECTED" ] && [ "$GITIGNORE_STATUS" = "ACTIVE" ] && [ "$DIRECTORY_PROTECTION" = "ACTIVE" ] && [ "$HOOK_STATUS" = "ACTIVE" ]; then
    echo "🎉 BULLETPROOF PROTECTION: FULLY ACTIVE"
    echo "🛡️  Your repository is completely protected against data loss!"
    OVERALL_STATUS=0
else
    echo "⚠️  PROTECTION: INCOMPLETE"
    echo "🔧 Some protection layers are missing or compromised."
    OVERALL_STATUS=1
fi

echo ""
echo "🚀 Safety check complete!"
echo ""

# Test protection by creating a test file
echo "🧪 Testing protection..."
mkdir -p outputs
echo "test" > outputs/test_protection_file.txt

if git status --porcelain | grep -q "outputs/test_protection_file.txt"; then
    echo "❌ PROTECTION TEST FAILED: Git is tracking files in outputs/"
    rm -f outputs/test_protection_file.txt
    exit 1
else
    echo "✅ PROTECTION TEST PASSED: Git properly ignores outputs/ files"
    rm -f outputs/test_protection_file.txt
fi

exit $OVERALL_STATUS





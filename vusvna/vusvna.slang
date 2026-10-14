/* vusvna: Vulkan Self-Validated Numerical Analysis.
 *
 * Affine arithmetic with guaranteed error bounds for Slang shaders, see README.md.
 * Add the vusvna directory of the repository (the directory of this file) to the include paths and
 * use `import vusvna;`.
 */
module vusvna;

__include "rounding.slang";
__include "interval.slang";
__include "approx.slang";
__include "aa.slang";
__include "real.slang";
__include "composite.slang";

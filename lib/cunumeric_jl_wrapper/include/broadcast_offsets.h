#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ufi {

struct WideOnlyOffsets {
  template <typename Descriptor>
  void observe(const Descriptor &) const {}
  void reject() const {}
};

// Conservative absolute element-offset bound, without signed overflow or
// overflowing intermediates. A singleton coordinate contributes no offset,
// even if its unused stride is outside Int32. Zero strides are likewise safe.
struct AdaptiveOffsets {
  bool eligible = true;

  void reject() { eligible = false; }

  template <typename Descriptor>
  void observe(const Descriptor &desc) {
    if (!eligible) return;
    constexpr auto limit = std::numeric_limits<std::int32_t>::max();
    std::int64_t remaining = limit;
    for (std::size_t d = 0; d < desc.dims.size(); ++d) {
      const auto dim = desc.dims[d];
      const auto stride = desc.strides[d];
      if (dim < 1 || dim > limit) return reject();
      if (dim == 1 || stride == 0) continue;
      if (stride == std::numeric_limits<std::int64_t>::min()) return reject();
      const auto magnitude = stride < 0 ? -stride : stride;
      const auto steps = dim - 1;
      if (magnitude > remaining / steps) return reject();
      remaining -= steps * magnitude;
    }
  }
};

template <typename Variant>
const Variant &select_broadcast_variant(const AdaptiveOffsets &offsets,
                                       const Variant &wide,
                                       const Variant &narrow) {
  return offsets.eligible ? narrow : wide;
}

}  // namespace ufi

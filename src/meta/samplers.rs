//! The AIR static-sampler global: its name, and the one order two independent scans must agree on.

/// Prefix of the module-scope global AIR emits for a `constexpr sampler`. A shader with several
/// distinct constexpr sampler states carries one global per state, uniqued by suffix.
pub const STATIC_SAMPLER_GLOBAL_PREFIX: &str = "__air_sampler_state";

/// Whether a module-scope global name is an AIR static-sampler state.
pub fn is_static_sampler_global(name: &str) -> bool {
    name.trim_start_matches('@')
        .starts_with(STATIC_SAMPLER_GLOBAL_PREFIX)
}

/// A TOTAL order over the static-sampler globals of one module.
///
/// Two independent scans put these globals in a sequence and then pair them off by position: the
/// interface pass, which collects them from the SPIR-V `OpName`s and hands each one the next free
/// sampler binding, and reflection, which collects them from the `!air.sampler_states` root and
/// reports the decoded state at each of those bindings. The two scans see different input orders,
/// so the sort has to be the whole of what decides the sequence -- a key that TIES leaves the rest
/// to input order, and the two sides then disagree about which state sits at which binding. Every
/// sample still gets a real sampler; it gets the other one's filter, address modes and compare
/// function, in a module that validates and binds cleanly.
///
/// LLVM uniques a repeated `__air_sampler_state` by appending `.N`, and re-uniques an already
/// uniqued name by appending again: `__air_sampler_state.163.1608`. Reading the suffix as a single
/// integer fails on that form and on the unsuffixed original alike, so both collapse to one key.
/// 14 of the 5499 sampler-state globals across 14579 local corpus sources carry the doubled form,
/// and 10 modules hold one beside the original.
///
/// Ordering by the whole dot-separated suffix cannot tie: a name is unique within a module, and it
/// is the last component of the key.
pub fn static_sampler_name_order(name: &str) -> (Vec<u64>, String) {
    let name = name.trim_start_matches('@');
    let components = name
        .strip_prefix(STATIC_SAMPLER_GLOBAL_PREFIX)
        .unwrap_or("")
        .split('.')
        .filter(|component| !component.is_empty())
        // A component that is not a number sorts after every one that is; the name settles it.
        .map(|component| component.parse::<u64>().unwrap_or(u64::MAX))
        .collect();
    (components, name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn re_uniqued_sampler_names_do_not_share_an_order_key() {
        let names = [
            "@__air_sampler_state",
            "@__air_sampler_state.118.9",
            "__air_sampler_state.119",
            "@__air_sampler_state.7",
        ];
        let mut sorted = names.to_vec();
        sorted.sort_by_key(|name| static_sampler_name_order(name));
        assert_eq!(
            sorted,
            [
                "@__air_sampler_state",
                "@__air_sampler_state.7",
                "@__air_sampler_state.118.9",
                "__air_sampler_state.119",
            ],
            "the whole dot-separated suffix orders these, and the leading `@` is not part of it"
        );
        let keys = names
            .iter()
            .map(|name| static_sampler_name_order(name))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            keys.len(),
            names.len(),
            "no two distinct sampler globals may share a key, or the order of the tied pair is \
             decided by whichever scan collected them"
        );
    }

    #[test]
    fn only_the_air_static_sampler_global_is_recognised() {
        assert!(is_static_sampler_global("@__air_sampler_state.118"));
        assert!(is_static_sampler_global("__air_sampler_state"));
        assert!(!is_static_sampler_global("@__air_sampler"));
        assert!(!is_static_sampler_global("@sampler_state"));
    }
}

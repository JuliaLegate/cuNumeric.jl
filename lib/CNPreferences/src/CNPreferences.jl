module CNPreferences
using Preferences
using LegatePreferences

const DEVEL_DEFAULT_WRAPPER_BRANCH = "main"

LegatePreferences.@make_preferences("cunumeric_")

end # module CNPreferences

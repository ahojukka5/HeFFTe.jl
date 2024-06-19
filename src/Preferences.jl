module Preferences

using Preferences

function set_heffte_path(new_path::String)
    # test that libheffte.so is found from new_path/lib
    if !isfile(joinpath(new_path, "lib", "libheffte.so"))
        throw(ArgumentError("libheffte.so not found in $(new_path)/lib"))
    end

    @set_preferences!("libheffte_path" => new_path)
    @info("Heffte path set; restart your Julia session for this change to take effect!")
end

const heffte_path = @load_preference("libheffte_path")

end # module

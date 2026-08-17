include(FetchContent)

find_package(OpenGL QUIET)

#=================== ImGui ===================
set(imgui_fixes_and_config_patch_file ${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/patches/imgui-fixes-and-config.patch)
set(imgui_apply_patch_command ${CMAKE_COMMAND} -Dpatch_file=${imgui_fixes_and_config_patch_file} -Dwith_reset=TRUE -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/git-patch.cmake)

FetchContent_Declare(
    ImGui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.9b-docking
    PATCH_COMMAND ${imgui_apply_patch_command}
)
FetchContent_MakeAvailable(ImGui)
list(APPEND ADDITIONAL_LIB_INCLUDES ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)

add_library(ImGui STATIC)
set_property(TARGET ImGui PROPERTY CXX_STANDARD 20)

target_sources(ImGui
    PRIVATE
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
)

target_sources(ImGui
    PRIVATE
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
)

target_include_directories(ImGui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends PRIVATE ${SDL2_INCLUDE_DIRS})

# ========= StormLib =============
if(INCLUDE_MPQ_SUPPORT)
    set(stormlib_patch_file ${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/patches/stormlib-optimizations.patch)
    set(stormlib_apply_patch_command ${CMAKE_COMMAND} -Dpatch_file=${stormlib_patch_file} -Dwith_reset=TRUE -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies/git-patch.cmake)

    FetchContent_Declare(
        StormLib
        GIT_REPOSITORY https://github.com/ladislav-zezula/StormLib.git
        GIT_TAG v9.25
        PATCH_COMMAND ${stormlib_apply_patch_command}
    )
    FetchContent_MakeAvailable(StormLib)
    list(APPEND ADDITIONAL_LIB_INCLUDES ${stormlib_SOURCE_DIR}/src)
endif()

#=================== STB ===================
set(STB_DIR ${CMAKE_BINARY_DIR}/_deps/stb)
# Checked, because an unchecked download here fails in a way that costs an hour to read back to.
#
# file(DOWNLOAD) without STATUS reports nothing: a refused, rate-limited or redirected request leaves a
# short file where the header should be, configuration continues, and every #include <stb_image.h> still
# resolves. The build then dies minutes later on "'stbi_image_free' was not declared in this scope" -- in a
# file that has included the header all along, which sends you looking anywhere but here.
#
# It also hides behind ccache. A translation unit that hits the cache never opens the header, so a bad
# download surfaces only in whichever files happen to miss, and the same tree builds or does not depending
# on what the cache holds. That is how this arrived: one file missed and the rest did not.
#
# The size test is the part that matters. An HTTP error page arrives with status 0 and is perfectly
# downloaded -- status alone would pass it. stb_image.h is around 280 KB, so anything under 100 KB is not
# the header whatever the transfer said.
file(DOWNLOAD "https://github.com/nothings/stb/raw/0bc88af4de5fb022db643c2d8e549a0927749354/stb_image.h"
     "${STB_DIR}/stb_image.h" STATUS STB_DOWNLOAD_STATUS TLS_VERIFY ON)
list(GET STB_DOWNLOAD_STATUS 0 STB_DOWNLOAD_CODE)
list(GET STB_DOWNLOAD_STATUS 1 STB_DOWNLOAD_MESSAGE)
if(NOT STB_DOWNLOAD_CODE EQUAL 0)
    # Removed on the way out for the same reason as below: a failed transfer still leaves a file, and a
    # leftover empty one would satisfy every #include on the next configure.
    file(REMOVE "${STB_DIR}/stb_image.h")
    message(FATAL_ERROR "Could not download stb_image.h: ${STB_DOWNLOAD_MESSAGE}")
endif()
file(SIZE "${STB_DIR}/stb_image.h" STB_HEADER_SIZE)
if(STB_HEADER_SIZE LESS 100000)
    file(REMOVE "${STB_DIR}/stb_image.h")
    message(FATAL_ERROR
        "stb_image.h downloaded but is only ${STB_HEADER_SIZE} bytes -- the request returned something "
        "other than the header (an error or redirect page). Removed it so the next configure retries.")
endif()
file(WRITE "${STB_DIR}/stb_impl.c" "#define STB_IMAGE_IMPLEMENTATION\n#include \"stb_image.h\"")

add_library(stb STATIC)

target_sources(stb PRIVATE
    ${STB_DIR}/stb_image.h
    ${STB_DIR}/stb_impl.c
)

target_include_directories(stb PUBLIC ${STB_DIR})
list(APPEND ADDITIONAL_LIB_INCLUDES ${STB_DIR})

#=================== libgfxd ===================
if (GFX_DEBUG_DISASSEMBLER)
    FetchContent_Declare(
        libgfxd
        GIT_REPOSITORY https://github.com/glankk/libgfxd.git
        GIT_TAG 008f73dca8ebc9151b205959b17773a19c5bd0da
    )
    FetchContent_MakeAvailable(libgfxd)

    add_library(libgfxd STATIC)
    set_property(TARGET libgfxd PROPERTY C_STANDARD 11)

    target_sources(libgfxd PRIVATE
        ${libgfxd_SOURCE_DIR}/gbi.h
        ${libgfxd_SOURCE_DIR}/gfxd.h
        ${libgfxd_SOURCE_DIR}/priv.h
        ${libgfxd_SOURCE_DIR}/gfxd.c
        ${libgfxd_SOURCE_DIR}/uc.c
        ${libgfxd_SOURCE_DIR}/uc_f3d.c
        ${libgfxd_SOURCE_DIR}/uc_f3db.c
        ${libgfxd_SOURCE_DIR}/uc_f3dex.c
        ${libgfxd_SOURCE_DIR}/uc_f3dex2.c
        ${libgfxd_SOURCE_DIR}/uc_f3dexb.c
    )

    target_include_directories(libgfxd PUBLIC ${libgfxd_SOURCE_DIR})
endif()

#======== thread-pool ========
FetchContent_Declare(
    ThreadPool
    GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
    GIT_TAG v4.1.0
)
FetchContent_MakeAvailable(ThreadPool)

list(APPEND ADDITIONAL_LIB_INCLUDES ${threadpool_SOURCE_DIR}/include)

#=========== prism ===========
option(PRISM_STANDALONE "Build prism as a standalone library" OFF)
FetchContent_Declare(
    prism
    GIT_REPOSITORY https://github.com/KiritoDv/prism-processor.git
    GIT_TAG bbcbc7e3f890a5806b579361e7aa0336acd547e7
)
FetchContent_MakeAvailable(prism)
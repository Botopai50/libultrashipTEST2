/*
 * Compiles one HLSL file to Shader Model 4 through D3DCompile, and prints whatever the compiler says.
 *
 * Exists for platforms without fxc. On Windows, use fxc from the Windows SDK instead -- it is the compiler
 * the game actually runs, and validate.sh prefers it when present. Elsewhere this is built with mingw and
 * run under Wine against Wine's d3dcompiler_47, which is a different implementation of the same profile:
 * good enough to catch undeclared identifiers, type errors and bad signatures, which is the class of
 * mistake that otherwise reaches a player as a crash. It is NOT authoritative about Shader-Model-4
 * limits -- see the README.
 *
 * D3DCompile is loaded by name rather than linked, for the same reason the renderer does it: it keeps the
 * build free of an import library for a DLL that is only present at run time.
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct ID3DBlob ID3DBlob;
typedef struct {
    void* QueryInterface;
    void* AddRef;
    void* Release;
    LPVOID(WINAPI* GetBufferPointer)(ID3DBlob*);
    SIZE_T(WINAPI* GetBufferSize)(ID3DBlob*);
} BlobVtbl;
struct ID3DBlob {
    const BlobVtbl* v;
};

typedef HRESULT(WINAPI* pD3DCompile)(const void*, SIZE_T, const char*, const void*, void*, const char*,
                                     const char*, UINT, UINT, ID3DBlob**, ID3DBlob**);

/* D3DCOMPILE_OPTIMIZATION_LEVEL2, matching the renderer's release build. */
#define OPT_LEVEL2 0xC000

int main(int argc, char** argv) {
    FILE* f;
    long n;
    char* buf;
    HMODULE mod;
    pD3DCompile compile;
    ID3DBlob* code = NULL;
    ID3DBlob* err = NULL;
    HRESULT hr;

    if (argc < 4) {
        printf("usage: hlslc <file.hlsl> <entrypoint> <profile>\n");
        return 2;
    }
    f = fopen(argv[1], "rb");
    if (!f) {
        printf("cannot open %s\n", argv[1]);
        return 2;
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (char*)malloc(n);
    if (!buf || fread(buf, 1, n, f) != (size_t)n) {
        printf("cannot read %s\n", argv[1]);
        return 2;
    }
    fclose(f);

    mod = LoadLibraryA("d3dcompiler_47.dll");
    if (!mod) {
        printf("d3dcompiler_47.dll not found\n");
        return 2;
    }
    compile = (pD3DCompile)GetProcAddress(mod, "D3DCompile");
    if (!compile) {
        printf("D3DCompile not exported\n");
        return 2;
    }

    hr = compile(buf, n, argv[1], NULL, NULL, argv[2], argv[3], OPT_LEVEL2, 0, &code, &err);
    if (err) {
        printf("%s\n", (char*)err->v->GetBufferPointer(err));
    }
    printf("[%s %s] %s\n", argv[2], argv[3], SUCCEEDED(hr) ? "OK" : "FAILED");
    return SUCCEEDED(hr) ? 0 : 1;
}

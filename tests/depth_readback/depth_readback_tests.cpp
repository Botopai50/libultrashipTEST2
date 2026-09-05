#include "fast/backends/gfx_direct3d11_depth_readback.h"
#include <iostream>
#include <stdexcept>
#include <thread>

using Microsoft::WRL::ComPtr;
using Fast::DepthReadbackDX11;

static void Check(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        Check(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0,
                                          D3D11_SDK_VERSION, &device, nullptr, &context)), "WARP device");
        const float first[] = { 0.25f, 0.75f };
        const float second[] = { 0.5f, 1.0f };
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = sizeof(first);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);
        D3D11_SUBRESOURCE_DATA initial{ first, 0, 0 };
        ComPtr<ID3D11Buffer> source;
        Check(SUCCEEDED(device->CreateBuffer(&desc, &initial, &source)), "source buffer");

        DepthReadbackDX11 reader;
        DepthReadbackDX11::Coordinates coordinates{ { 10.0f, 20.0f }, { 30.0f, 40.0f } };
        std::vector<float> values;
        auto now = DepthReadbackDX11::Clock::now();
        auto read = [&](bool deferred = true) {
            Check(SUCCEEDED(reader.Read(device.Get(), context.Get(), source.Get(), coordinates,
                                        deferred, values, now)), "readback result");
        };
        read();
        Check(values == std::vector<float>(first, first + 2), "cold read must use current values");
        context->UpdateSubresource(source.Get(), 0, nullptr, second, 0, 0);
        read();
        Check(values == std::vector<float>(first, first + 2), "stationary query should reuse previous values");

        // Finish submitted GPU work in the TEST only. The implementation polls once with DONOTFLUSH
        // and maps completed slots with DO_NOT_WAIT; it never spins or flushes to collect them.
        D3D11_QUERY_DESC queryDesc{ D3D11_QUERY_EVENT, 0 };
        ComPtr<ID3D11Query> complete;
        Check(SUCCEEDED(device->CreateQuery(&queryDesc, &complete)), "completion query");
        context->End(complete.Get());
        context->Flush();
        const auto deadline = DepthReadbackDX11::Clock::now() + std::chrono::seconds(5);
        HRESULT ready;
        while ((ready = context->GetData(complete.Get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH)) == S_FALSE &&
               DepthReadbackDX11::Clock::now() < deadline) {
            std::this_thread::yield();
        }
        Check(ready == S_OK, "GPU completion timeout");
        context->UpdateSubresource(source.Get(), 0, nullptr, first, 0, 0);
        read();
        Check(values == std::vector<float>(second, second + 2), "completed ring copy should advance result");

        // Same count, different positions: never associate by array index alone.
        coordinates[0].first += 1;
        read();
        Check(values == std::vector<float>(first, first + 2), "moving positions require current data");
        context->UpdateSubresource(source.Get(), 0, nullptr, second, 0, 0);
        now += std::chrono::seconds(1);
        read();
        Check(values == std::vector<float>(second, second + 2), "stale result must expire");

        context->UpdateSubresource(source.Get(), 0, nullptr, first, 0, 0);
        reader.Reset();
        read();
        Check(values == std::vector<float>(first, first + 2), "source reset must discard history");
        context->UpdateSubresource(source.Get(), 0, nullptr, second, 0, 0);
        read(false);
        Check(values == std::vector<float>(second, second + 2), "disabled mode must use current data");
        coordinates.pop_back();
        read();
        Check(values.size() == 1 && values[0] == second[0], "smaller batch must not reuse old layout");
        coordinates.clear();
        read();
        Check(values.empty(), "empty batch");
        coordinates = { { 0.0f, 0.0f }, { 1.0f, 1.0f }, { 2.0f, 2.0f } };
        Check(reader.Read(device.Get(), context.Get(), source.Get(), coordinates, true, values, now) == E_INVALIDARG,
              "oversized batch must be rejected");
        std::cout << "PASS: WARP readback, deferred ring, position matching, expiry, reset, sync and batch sizes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

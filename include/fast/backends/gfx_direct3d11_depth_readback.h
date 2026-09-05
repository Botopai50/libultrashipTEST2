#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace Fast {

// Adaptive readback for visual occlusion only. Reuse is limited to the immediately preceding query
// with exactly the same coordinates. A changed layout, expired result or cold cache reads synchronously.
// The owner must Reset() when the depth source, framebuffer dimensions or MSAA configuration changes.
class DepthReadbackDX11 {
  public:
    using Coordinates = std::vector<std::pair<float, float>>;
    using Clock = std::chrono::steady_clock;

    void Reset() {
        mSlots = {};
        mSynchronous.Reset();
        mByteWidth = 0;
        mSequence = 0;
        mCachedSequence = 0;
        mCachedCoordinates.clear();
        mCachedValues.clear();
    }

    HRESULT Read(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Buffer* source,
                 const Coordinates& coordinates, bool deferred, std::vector<float>& values,
                 Clock::time_point now = Clock::now()) {
        values.clear();
        if (coordinates.empty()) {
            return S_OK;
        }
        D3D11_BUFFER_DESC desc{};
        source->GetDesc(&desc);
        if (coordinates.size() > desc.ByteWidth / sizeof(float)) {
            return E_INVALIDARG;
        }
        if (mByteWidth != desc.ByteWidth) {
            Reset();
            mByteWidth = desc.ByteWidth;
        }
        ++mSequence;

        if (deferred) {
            for (auto& slot : mSlots) {
                if (!slot.pending) {
                    continue;
                }
                const HRESULT ready = context->GetData(slot.complete.Get(), nullptr, 0,
                                                       D3D11_ASYNC_GETDATA_DONOTFLUSH);
                if (ready == S_FALSE) {
                    continue;
                }
                if (FAILED(ready)) {
                    return ready;
                }
                D3D11_MAPPED_SUBRESOURCE mapped{};
                const HRESULT hr = context->Map(slot.buffer.Get(), 0, D3D11_MAP_READ,
                                                D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped);
                if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
                    continue;
                }
                if (FAILED(hr)) {
                    return hr;
                }
                if (slot.sequence > mCachedSequence) {
                    mCachedCoordinates = slot.coordinates;
                    mCachedValues.resize(slot.coordinates.size());
                    std::memcpy(mCachedValues.data(), mapped.pData, mCachedValues.size() * sizeof(float));
                    mCachedSequence = slot.sequence;
                    mCachedAt = slot.submitted;
                }
                context->Unmap(slot.buffer.Get(), 0);
                slot.pending = false;
            }

            if (mCachedSequence + 1 == mSequence && mCachedCoordinates == coordinates &&
                now - mCachedAt <= std::chrono::milliseconds(100)) {
                values = mCachedValues;
                // Never overwrite a pending copy and never wait for a slot. If all slots are busy,
                // the next query falls back to the synchronous path once this cached result expires.
                for (auto& slot : mSlots) {
                    if (slot.pending) {
                        continue;
                    }
                    HRESULT hr = EnsureBuffer(device, desc, slot.buffer);
                    if (FAILED(hr)) {
                        return hr;
                    }
                    if (!slot.complete) {
                        const D3D11_QUERY_DESC query{ D3D11_QUERY_EVENT, 0 };
                        hr = device->CreateQuery(&query, slot.complete.GetAddressOf());
                        if (FAILED(hr)) {
                            return hr;
                        }
                    }
                    slot.coordinates = coordinates;
                    slot.sequence = mSequence;
                    slot.submitted = now;
                    context->CopyResource(slot.buffer.Get(), source);
                    context->End(slot.complete.Get());
                    slot.pending = true;
                    break;
                }
                return S_OK;
            }
        }

        HRESULT hr = EnsureBuffer(device, desc, mSynchronous);
        if (FAILED(hr)) {
            return hr;
        }
        context->CopyResource(mSynchronous.Get(), source);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = context->Map(mSynchronous.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) {
            return hr;
        }
        values.resize(coordinates.size());
        std::memcpy(values.data(), mapped.pData, values.size() * sizeof(float));
        context->Unmap(mSynchronous.Get(), 0);
        mCachedCoordinates = coordinates;
        mCachedValues = values;
        mCachedSequence = mSequence;
        mCachedAt = now;
        return S_OK;
    }

  private:
    static HRESULT EnsureBuffer(ID3D11Device* device, D3D11_BUFFER_DESC desc,
                                Microsoft::WRL::ComPtr<ID3D11Buffer>& buffer) {
        if (buffer) {
            return S_OK;
        }
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.MiscFlags = 0;
        desc.StructureByteStride = 0;
        return device->CreateBuffer(&desc, nullptr, buffer.GetAddressOf());
    }

    struct Slot {
        Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
        Microsoft::WRL::ComPtr<ID3D11Query> complete;
        Coordinates coordinates;
        uint64_t sequence = 0;
        Clock::time_point submitted{};
        bool pending = false;
    };
    std::array<Slot, 3> mSlots;
    Microsoft::WRL::ComPtr<ID3D11Buffer> mSynchronous;
    UINT mByteWidth = 0;
    uint64_t mSequence = 0;
    uint64_t mCachedSequence = 0;
    Clock::time_point mCachedAt{};
    Coordinates mCachedCoordinates;
    std::vector<float> mCachedValues;
};

} // namespace Fast

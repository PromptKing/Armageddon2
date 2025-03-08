#include "pch.h"
#include "VoiceModel.h"
#if __has_include("VoiceModel.g.cpp")
#include "VoiceModel.g.cpp"
#endif

using namespace winrt;
using namespace Windows::UI::Xaml;

namespace winrt::Armageddon2::implementation
{
    int32_t VoiceModel::MyProperty()
    {
        throw hresult_not_implemented();
    }

    void VoiceModel::MyProperty(int32_t /* value */)
    {
        throw hresult_not_implemented();
    }

    void VoiceModel::ClickHandler(IInspectable const&, RoutedEventArgs const&)
    {
        Button().Content(box_value(L"Clicked"));
    }
}

#pragma once

#include "VoiceModel.g.h"

namespace winrt::Armageddon2::implementation
{
    struct VoiceModel : VoiceModelT<VoiceModel>
    {
        VoiceModel() 
        {
            // Xaml objects should not call InitializeComponent during construction.
            // See https://github.com/microsoft/cppwinrt/tree/master/nuget#initializecomponent
        }

        int32_t MyProperty();
        void MyProperty(int32_t value);

        void ClickHandler(Windows::Foundation::IInspectable const& sender, Windows::UI::Xaml::RoutedEventArgs const& args);
    };
}

namespace winrt::Armageddon2::factory_implementation
{
    struct VoiceModel : VoiceModelT<VoiceModel, implementation::VoiceModel>
    {
    };
}

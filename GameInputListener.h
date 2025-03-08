#ifndef GAME_INPUT_LISTENER_H
#define GAME_INPUT_LISTENER_H

#include <windows.h>
#include <Xinput.h>
#include <GameInput.h>
#include <thread>

// Link XInput Library
#pragma comment(lib, "Xinput.lib")

class GameInputListener {
public:
    GameInputListener();
    ~GameInputListener();

    void StartListener();
    void StopListener();

private:
    IGameInput* gameInput;
    bool running;

    void ListenForDXInit();
    void ProcessGameInput(uintptr_t gameInputBase, uintptr_t gameInputInboxBase, uintptr_t xInputBase, uintptr_t hidBase);
    void PredictJoystickMovement();
    void MapButtonHistory();
    void WritePredictionsToMemory(uintptr_t gameInputBase, uintptr_t gameInputInboxBase, uintptr_t xInputBase, uintptr_t hidBase, GameInputGamepadState& state);

    uintptr_t GetModuleBaseAddress(const char* moduleName);
};

#endif // GAME_INPUT_LISTENER_H

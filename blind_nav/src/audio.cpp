#include "audio.h"
#include <cstdlib>
#include <string>

void speak(const std::string& text) {
    // -v ru: голос, -s 150: скорость, 2>/dev/null: убираем спам ALSA
    std::string cmd = "espeak-ng -v ru \"" + text + "\" -s 150 > /dev/null 2>&1 &"; 
    std::system(cmd.c_str());
}

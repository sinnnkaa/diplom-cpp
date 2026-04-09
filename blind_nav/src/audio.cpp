#include "audio.h"
#include <cstdlib>
#include <string>

void speak(const std::string& text) {
    std::string cmd = "espeak-ng -v ru \"" + text + "\" -s 150 > /dev/null 2>&1 &"; 
    std::system(cmd.c_str());
}

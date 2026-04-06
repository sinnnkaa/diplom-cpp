#include <cstdlib>
#include <string>

void speak(const std::string& text) {
    std::string cmd = "espeak \"" + text + "\"";
    system(cmd.c_str());
}
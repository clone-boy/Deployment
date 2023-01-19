#include"net/utils.h"
#include<stdarg.h>
#include<memory>
#include<chrono>
#include"net/net.h"
#include<map>
#include<functional>
#include<signal.h>
#include<string.h>

std::string utils::format(const char *fmt, ...) {
    char buffer[500];
    std::unique_ptr<char[]> release1;
    char *base;
    for (int iter = 0; iter < 2; iter++) {
        int bufsize;
        if (iter == 0) {
            bufsize = sizeof(buffer);
            base = buffer;
        } else {
            bufsize = 10000;
            base = new char[bufsize];
            release1.reset(base);
        }
        char *p = base;
        char *limit = base + bufsize;
        if (p < limit) {
            va_list ap;
            va_start(ap, fmt);
            p += vsnprintf(p, limit - p, fmt, ap);
            va_end(ap);
        }
        // Truncate to available space if necessary
        if (p >= limit) {
            if (iter == 0) {
                continue;  // Try again with larger buffer
            } else {
                p = limit - 1;
                *p = '\0';
            }
        }
        break;
    }
    return base;
}


int64_t utils::steadyMilli() {
    std::chrono::time_point<std::chrono::steady_clock> p = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(p.time_since_epoch()).count();
}


static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string utils::base64_encode(const char *bytes_to_encode, unsigned int in_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];  // store 3 byte of bytes_to_encode
    unsigned char char_array_4[4];  // store encoded character to 4 bytes

    while(in_len--){
        char_array_3[i++] = *(bytes_to_encode++);  // get three bytes (24 bits)
        if (i == 3) {
            // eg. we have 3 bytes as ( 0100 1101, 0110 0001, 0110 1110) --> (010011, 010110, 000101, 101110)
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2; // get first 6 bits of first byte,
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4); // get last 2 bits of first byte and first 4 bit of second byte
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6); // get last 4 bits of second byte and first 2 bits of third byte
            char_array_4[3] = char_array_3[2] & 0x3f; // get last 6 bits of third byte

            for (i = 0; (i < 4); i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i){
        for (j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while ((i++ < 3))
            ret += '=';
    }

    return ret;

}

std::string utils::base64_decode(const std::string& encoded_string) {
    size_t in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    while(in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])){
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]) & 0xff;

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret += char_array_3[i];
            i = 0;
        }
    }

    if(i){
        for (j = 0; j < i; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]) & 0xff;

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

        for (j = 0; (j < i - 1); j++)
            ret += char_array_3[j];
    }

    return ret;
}



int LineCodec::tryDecode(Slice data, Slice &msg) {
    if (data.size() == 1 && data[0] == 0x04) {
        msg = data;
        return 1;
    }
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] == '\n') {
            if (i > 0 && data[i - 1] == '\r') {
                msg = Slice(data.data(), i - 1);
                return static_cast<int>(i + 1);
            } else {
                msg = Slice(data.data(), i);
                return static_cast<int>(i + 1);
            }
        }
    }
    return 0;
}
void LineCodec::encode(Slice msg, Buffer &buf) {
    buf.append(msg).append("\r\n");
}

int LengthCodec::tryDecode(Slice data, Slice &msg) {
    if (data.size() < 8) {
        return 0;
    }
    int len = net::ntoh(*(int32_t *) (data.data() + 4));
    if (len > 1024 * 1024 || memcmp(data.data(), "mBdT", 4) != 0) {
        return -1;
    }
    if ((int) data.size() >= len + 8) {
        msg = Slice(data.data() + 8, len);
        return len + 8;
    }
    return 0;
}
void LengthCodec::encode(Slice msg, Buffer &buf) {
    buf.append("mBdT").appendValue(net::hton((int32_t) msg.size())).append(msg);
}


std::map<int, std::function<void()>> handlers;

void signal_handler(int sig) {
    handlers[sig]();
}

void Signal::signal(int sig, const std::function<void()> &handler) {
    handlers[sig] = handler;
    ::signal(sig, signal_handler);
}
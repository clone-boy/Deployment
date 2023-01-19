#ifndef UTILS_H
#define UTILS_H
#include<string>
#include<netinet/in.h>
#include"net/buffer.h"
#include<functional>

class noncopyable {
protected:
	noncopyable() = default;
	~noncopyable() = default;
private:
	noncopyable(const noncopyable&) = delete;
	noncopyable& operator=( const noncopyable& ) = delete;
};

class utils{
public:
    static std::string format(const char* fmt, ...);
    static int64_t steadyMilli();
    static std::string base64_encode(const char *bytes_to_encode, unsigned int in_len);
    static std::string base64_decode(const std::string& encoded_string);
    static inline bool is_base64(unsigned char c) {
        return (isalnum(c) || (c == '+') || (c == '/'));
    }
};


class CodecBase {
public:
    // > 0 解析出完整消息，消息放在msg中，返回已扫描的字节数
    // == 0 解析部分消息
    // < 0 解析错误
    virtual int tryDecode(Slice data, Slice &msg) = 0;
    virtual void encode(Slice msg, Buffer &buf) = 0;
    virtual CodecBase *clone() = 0;
    virtual ~CodecBase() = default;
};

//以\r\n结尾的消息
class LineCodec : public CodecBase {
public:
    int tryDecode(Slice data, Slice &msg) override;
    void encode(Slice msg, Buffer &buf) override;
    CodecBase *clone() override { return new LineCodec(); }
};

//给出长度的消息
class LengthCodec : public CodecBase {
public:
    int tryDecode(Slice data, Slice &msg) override;
    void encode(Slice msg, Buffer &buf) override;
    CodecBase *clone() override { return new LengthCodec(); }
};


class Signal {
public:
    static void signal(int sig, const std::function<void()> &handler);
};

#endif
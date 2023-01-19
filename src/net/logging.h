#ifndef LOGGING_H
#define LOGGING_H

#include<stdio.h>
#include<unistd.h>
#include"net/utils.h"

#define hlog(level, ...)                                                                \
    do {                                                                                \
        if (level <= Logger::getLogger().getLogLevel()) {                               \
            Logger::getLogger().logv(level, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        }                                                                               \
    } while (0)


#define trace(...) hlog(Logger::LTRACE, __VA_ARGS__)
#define debug(...) hlog(Logger::LDEBUG, __VA_ARGS__)
#define info(...) hlog(Logger::LINFO, __VA_ARGS__)
#define warn(...) hlog(Logger::LWARN, __VA_ARGS__)
#define error(...) hlog(Logger::LERROR, __VA_ARGS__)
#define fatal(...) hlog(Logger::LFATAL, __VA_ARGS__)
#define fatalif(b, ...)                        \
    do {                                       \
        if ((b)) {                             \
            hlog(Logger::LFATAL, __VA_ARGS__); \
        }                                      \
    } while (0)
//assert
#define check(b, ...)                          \
    do {                                       \
        if (!(b)) {                             \
            hlog(Logger::LFATAL, __VA_ARGS__); \
        }                                      \
    } while (0)
#define exitif(b, ...)                         \
    do {                                       \
        if ((b)) {                             \
            hlog(Logger::LERROR, __VA_ARGS__); \
            _exit(1);                          \
        }                                      \
    } while (0)

#define setloglevel(l) Logger::getLogger().setLogLevel(l)
#define setlogfile(n) Logger::getLogger().setFileName(n)

class Logger: private noncopyable{
public:
    enum LogLevel { LFATAL = 0, LERROR, LUERR, LWARN, LINFO, LDEBUG, LTRACE, LALL };
    Logger();
    ~Logger();
    static Logger& getLogger();
    LogLevel getLogLevel() { return level_; }
    void setFileName(const std::string &filename);
    void setLogLevel(const std::string &level);
    void setLogLevel(LogLevel level){level_ = std::min(LALL, std::max(level, LFATAL));}
    void logv(int level, const char *file, int line, const char *func, const char *fmt, ...);
private:
    static const char *levelStrs_[LALL + 1];
    LogLevel level_;
    int fd_; //-1 for strerr, >0 for file
    std::string filename_;
};





#endif
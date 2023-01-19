#include"net/logging.h"
#include<time.h>
#include<stdarg.h>
#include<sys/time.h>
#include<string.h>
#include<assert.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>

Logger::Logger():level_(LINFO){
    tzset();//时区设置
    fd_=-1;
}

Logger::~Logger(){
    if (fd_ != -1) {
        close(fd_);
    }
}

Logger& Logger::getLogger(){
    static Logger logger;
    return logger;
}

const char *Logger::levelStrs_[LALL + 1] = {
    "FATAL", "ERROR", "UERR", "WARN", "INFO", "DEBUG", "TRACE", "ALL"
};

void Logger::setFileName(const std::string &filename){
    int fd = open(filename.c_str(), O_APPEND | O_CREAT | O_WRONLY | O_CLOEXEC, DEFFILEMODE);
    if (fd < 0) {
        fprintf(stderr, "open log file %s failed. msg: %s ignored\n", filename.c_str(), strerror(errno));
        return;
    }
    filename_ = filename;
    if (fd_ == -1) {
        fd_ = fd;
    } else {
        int r = dup2(fd, fd_);
        fatalif(r < 0, "dup2 failed");
        close(fd);
    }
}

void Logger::setLogLevel(const std::string &level){
    LogLevel ilevel = LINFO;
    for (size_t i = 0; i < sizeof(levelStrs_) / sizeof(const char *); i++) {
        if (strcasecmp(levelStrs_[i], level.c_str()) == 0) {
            ilevel = (LogLevel) i;
            break;
        }
    }
    setLogLevel(ilevel);
}

void Logger::logv(int level, const char *file, int line, const char *func, const char *fmt, ...){
    if (level > level_) {
        return;
    }
    char buffer[4 * 1024];
    char *p = buffer;
    char *limit = buffer + sizeof(buffer);

    struct timeval now_tv;
    gettimeofday(&now_tv, NULL);
    const time_t seconds = now_tv.tv_sec;
    struct tm t;
    localtime_r(&seconds, &t);
    p += snprintf(p, limit - p, "%04d/%02d/%02d-%02d:%02d:%02d.%06d %s %s:%d ", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec,
                  static_cast<int>(now_tv.tv_usec), levelStrs_[level], file, line);
    va_list args;
    va_start(args, fmt);
    p += vsnprintf(p, limit - p, fmt, args);
    va_end(args);
    p = std::min(p, limit - 2);
    // trim the ending \n
    while (*--p == '\n') {
    }
    *++p = '\n';
    *++p = '\0';
    int fd = fd_ == -1 ? 1 : fd_;
    int err = ::write(fd, buffer, p - buffer);
    if (err != p - buffer) {
        fprintf(stderr, "write log file %s failed. written %d errmsg: %s\n", filename_.c_str(), err, strerror(errno));
    }
    if (level == LFATAL) {
        fprintf(stderr, "%s", buffer);
        assert(0);
    }
}
#include"net/net.h"
#include"net/utils.h"
#include<string.h>
#include<iostream>
#include<stdio.h>
#include"net/logging.h"
#include<netdb.h>
#include<fcntl.h>


IPv4Addr::IPv4Addr(const std::string& host, unsigned short port){
    memset(&addr_,0,sizeof(addr_));
    addr_.sin_family = AF_INET;
    addr_.sin_port = htons(port); //big endian
    if(host.size())
        addr_.sin_addr = getHostByName(host); //big endian
    else
        addr_.sin_addr.s_addr = INADDR_ANY;
    if(addr_.sin_addr.s_addr == INADDR_NONE)
        error("cannot resolve %s to ip", host.c_str());
}

std::string IPv4Addr::addrToString() const{
    uint32_t addr = addr_.sin_addr.s_addr;
    return utils::format("%d.%d.%d.%d:%d", (addr>>0)&0xff,(addr>>8)&0xff,(addr>>16)&0xff,(addr>>24)&0xff,ntohs(addr_.sin_port));
}
std::string IPv4Addr::ipToString() const{
    uint32_t addr = addr_.sin_addr.s_addr;
    return utils::format("%d.%d.%d.%d", (addr>>0)&0xff,(addr>>8)&0xff,(addr>>16)&0xff,(addr>>24)&0xff);
}
unsigned short IPv4Addr::port() const{
    return (unsigned short) ntohs(addr_.sin_port);
}

unsigned int IPv4Addr::ipInt() const{
    return (unsigned int) ntohl(addr_.sin_addr.s_addr);
}

struct in_addr IPv4Addr::getHostByName(const std::string &host){
    struct in_addr addr;
    struct hostent* he = gethostbyname(host.c_str());
    if (he && he->h_addrtype == AF_INET) {
        addr = *reinterpret_cast<struct in_addr *>(he->h_addr);
    } else {
        addr.s_addr = INADDR_NONE;
    }
    return addr;
}

int net::setNonBlock(int fd, bool value){
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) {
        return errno;
    }
    if (value) {
        return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }
    return fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
}

int net::addFdFlag(int fd, int flag){
    int ret = fcntl(fd, F_GETFL, 0);
    return fcntl(fd, F_SETFL, ret | flag);
}

int net::setReuseAddr(int fd, bool value) {
    int flag = value;
    int len = sizeof flag;
    return setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &flag, len);
}

int net::setReusePort(int fd, bool value) {
#ifndef SO_REUSEPORT
    fatalif(value, "SO_REUSEPORT not supported");
    return 0;
#else
    int flag = value;
    int len = sizeof flag;
    return setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &flag, len);
#endif
}
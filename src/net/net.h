#ifndef NET_H
#define NET_H

#include<string>
#include<netinet/in.h>

static const int kLittleEndian = LITTLE_ENDIAN;
inline uint16_t htobe(uint16_t v) {
	if (!kLittleEndian) {
		return v;
	}
	unsigned char *pv = (unsigned char *) &v;
	return uint16_t(pv[0]) << 8 | uint16_t(pv[1]);
}
inline uint32_t htobe(uint32_t v) {
	if (!kLittleEndian) {
		return v;
	}
	unsigned char *pv = (unsigned char *) &v;
	return uint32_t(pv[0]) << 24 | uint32_t(pv[1]) << 16 | uint32_t(pv[2]) << 8 | uint32_t(pv[3]);
}
inline uint64_t htobe(uint64_t v) {
	if (!kLittleEndian) {
		return v;
	}
	unsigned char *pv = (unsigned char *) &v;
	return uint64_t(pv[0]) << 56 | uint64_t(pv[1]) << 48 | uint64_t(pv[2]) << 40 | uint64_t(pv[3]) << 32 | uint64_t(pv[4]) << 24 | uint64_t(pv[5]) << 16 |
		uint64_t(pv[6]) << 8 | uint64_t(pv[7]);
}
inline int16_t htobe(int16_t v) {
	return (int16_t) htobe((uint16_t) v);
}
inline int32_t htobe(int32_t v) {
	return (int32_t) htobe((uint32_t) v);
}
inline int64_t htobe(int64_t v) {
	return (int64_t) htobe((uint64_t) v);
}



class IPv4Addr {
private:
    struct sockaddr_in addr_;
public:
    IPv4Addr(const std::string& host, unsigned short port);
    IPv4Addr(unsigned short port=0):IPv4Addr("", port){}
    IPv4Addr(const struct sockaddr_in& addr):addr_(addr){}
    IPv4Addr(const IPv4Addr& addr){addr_ = addr.addr_;}
    std::string addrToString() const;
    std::string ipToString() const;
    unsigned short port() const;
    unsigned int ipInt() const;
    struct sockaddr_in &getAddr() {
        return addr_;
    }
    static std::string hostToIp(const std::string &host) {
        IPv4Addr addr(host, 0);
        return addr.ipToString();
    }
    static struct in_addr getHostByName(const std::string &host);
};

class net
{
public:
    template <class T>
    static T hton(T v) {
        return htobe(v);
    }
    template <class T>
    static T ntoh(T v) {
        return htobe(v);
    }
    static int setNonBlock(int fd, bool value = true);
    static int addFdFlag(int fd, int flag);
    static int setReuseAddr(int fd, bool value = true);
    static int setReusePort(int fd, bool value = true);
};




#endif
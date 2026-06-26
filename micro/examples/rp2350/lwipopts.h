// Minimal lwIP configuration for the voice WiFi-setup app (moonshine_micro_echo_wifi).
//
// This firmware only needs to ASSOCIATE with an AP, obtain a DHCP lease, and
// read back the assigned IPv4 address -- it never opens a socket. So this is
// tuned for the smallest possible footprint on the RAM-constrained RP2350,
// where the SpellingCNN tensor arena already claims the bulk of SRAM:
//   * NO_SYS / poll mode (no RTOS, no threads) -- matches pico_cyw43_arch_lwip_poll.
//   * TCP, DNS, sockets, and the netconn API are all compiled out.
//   * Small pbuf pool / mem heap, since DHCP exchanges only a handful of small
//     UDP datagrams.
//
// If a future revision needs an actual network client (NTP, HTTP, ...), bump
// MEM_SIZE / PBUF_POOL_SIZE and re-enable LWIP_TCP / LWIP_DNS, and re-check the
// RAM budget against the arena.

#ifndef SPELLING_LWIPOPTS_H_
#define SPELLING_LWIPOPTS_H_

// --- Core model: bare-metal, polled (no RTOS) ------------------------------
#define NO_SYS                      1
#define LWIP_SOCKET                 0
#define LWIP_NETCONN                0
#define SYS_LIGHTWEIGHT_PROT        0

// --- Memory: use lwIP's own pools (not libc malloc), kept small ------------
#define MEM_LIBC_MALLOC             0
#define MEM_ALIGNMENT               4
#define MEM_SIZE                    4000
#define MEMP_NUM_PBUF               8
#define MEMP_NUM_UDP_PCB            4
#define MEMP_NUM_ARP_QUEUE          2
#define MEMP_NUM_SYS_TIMEOUT        8
#define PBUF_POOL_SIZE              8
#define PBUF_POOL_BUFSIZE           1536

// --- Protocol selection: just enough for WPA2 join + DHCP ------------------
#define LWIP_IPV4                   1
#define LWIP_IPV6                   0
#define LWIP_ARP                    1
#define LWIP_ETHERNET               1
#define LWIP_ICMP                   1
#define LWIP_RAW                    0
#define LWIP_UDP                    1
#define LWIP_TCP                    0
#define LWIP_DNS                    0
#define LWIP_DHCP                   1
#define DHCP_DOES_ARP_CHECK         0
#define LWIP_DHCP_DOES_ACD_CHECK    0

// --- netif callbacks the cyw43 driver uses to track link/address state -----
#define LWIP_NETIF_STATUS_CALLBACK  1
#define LWIP_NETIF_LINK_CALLBACK    1
#define LWIP_NETIF_HOSTNAME         1
#define LWIP_NETIF_TX_SINGLE_PBUF   1

// --- Checksums: let lwIP compute them (the cyw43 path has no HW offload) ----
#define LWIP_CHKSUM_ALGORITHM       3

// --- Drop all stats / debug to save flash + RAM ----------------------------
#define LWIP_STATS                  0
#define MEM_STATS                   0
#define SYS_STATS                   0
#define MEMP_STATS                  0
#define LINK_STATS                  0
#define ETHARP_STATS                0
#define IP_STATS                    0
#define UDP_STATS                   0
#define ICMP_STATS                  0

#ifndef NDEBUG
#define LWIP_DEBUG                  1
#endif
#define LWIP_STATS_DISPLAY          0

#endif  // SPELLING_LWIPOPTS_H_

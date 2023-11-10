from scapy.all import srp, Ether, ARP
import socket 

try:
 ip= "192.168.20.9"   
 arp= Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(op=1, pdst=ip)
 ans, _= srp(arp, timeout=2, verbose=False)
 devices= [{"ip": response[1].psrc, "mac": response[1].hwsrc} for response in ans]

# hostname, _, _= socket.gethostbyaddr(ip)
# if hostname is not None:
#     print(f"this is name {hostname} of ip {ip}")
# else:
#     print(f"no encontro hostname of ip {ip}")    

 print({"success":True, "devices": devices})
except Exception as e:
    print(e)
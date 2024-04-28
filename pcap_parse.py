# -*- coding: UTF-8 -*-
import os

import dpkt
from ipaddress import ip_address

import numpy as np
from tqdm import tqdm
from io import StringIO

import gzip
import json

MAX_LOAD = 28*28

def get_BD(payload):
    BD = np.zeros(shape=(0x100,),dtype=np.int_)
    for byte in payload:
        BD[byte] += 1
    return BD


def get_IP(buf):
    try:
        ipv4 = dpkt.ip.IP(buf)
        return ipv4
    except:
        pass
    try:
        ipv6 = dpkt.ip6.IP6(buf)
        return ipv6
    except:
        return None


def pcap_replay(pcapfile, processbar=True):
    FLOWs = {}
    with open(pcapfile, 'rb') as f:
        try:
            pkts = dpkt.pcap.Reader(f).readpkts()
        except:
            print(f"Error Reading {pcapfile}")
            return FLOWs

    if processbar:
        pbar = tqdm(pkts, desc=f'Replay {pcapfile}', unit='Pkt', unit_scale=True)
    else:
        pbar = pkts

    for idx, (ts, buf) in enumerate(pbar):
        ip = get_IP(buf)
        if isinstance(ip, dpkt.ip.IP):
            proto = ip.p
        elif isinstance(ip, dpkt.ip6.IP6):
            proto = ip.nxt
        else:
            continue
        srcIP, dstIP = ip_address(ip.src).exploded, ip_address(ip.dst).exploded
        # print(srcIP,dstIP)

        if proto == 6:  # TCP
            tcp = ip.data
            srcPort, dstPort = tcp.sport, tcp.dport
            payload = tcp.data
        elif proto == 17:
            udp = ip.data
            srcPort, dstPort = udp.sport, udp.dport
            payload = udp.data
        else:
            # print("Not TCP/UDP packet")
            continue

        # print(idx+1,f"{srcIP}:{srcPort}->{dstIP}:{dstPort} {proto}")
        key = (srcIP, dstIP, srcPort, dstPort, proto)
        key_ = (dstIP, srcIP, dstPort, srcPort, proto)
        if key not in FLOWs.keys() and key_ not in FLOWs.keys():
            p = {'t':0., 'd':'>', 'l':len(buf), 'b':len(payload), 'iat':0. }
            FLOWs[key] = {
                "ts": ts,
                "bytes_out": len(payload), "pkts_out": 1,
                "bytes_in": 0, "pkts_in": 0,
                "packets": [p],
                'upload': [],
                'download': [],
                'upBD': np.zeros(shape=(0x100,),dtype=np.int_),
                'downBD': np.zeros(shape=(0x100,),dtype=np.int_),

                "last_up_p": ts,
                "last_down_p": None
            }
            FLOWs[key]['upBD'] += get_BD(payload)
            if len(FLOWs[key]['upload']) < MAX_LOAD:
                FLOWs[key]['upload'] += list(payload[0:MAX_LOAD - len(FLOWs[key]['upload'])])
        else:
            if key in FLOWs.keys(): # The packet is in the upflow
                ts0 = FLOWs[key]["ts"] # start time of the flow
                p = {'t':ts-ts0, 'd':'>', 'l':len(buf), 'b':len(payload), 'iat':ts-FLOWs[key]["last_up_p"]}
                FLOWs[key]["last_up_p"] = ts

                FLOWs[key]["packets"].append(p)
                FLOWs[key]["bytes_out"] += len(payload)
                FLOWs[key]["pkts_out"] += 1

                FLOWs[key]['upBD'] += get_BD(payload)
                if len(FLOWs[key]['upload']) < MAX_LOAD:
                    FLOWs[key]['upload'] += list(payload[0:MAX_LOAD-len(FLOWs[key]['upload'])])
            elif key_ in FLOWs.keys(): # The packet is in the downflow
                if FLOWs[key_]["last_down_p"]==None:
                    FLOWs[key_]["last_down_p"] = ts
                ts0 = FLOWs[key_]["ts"]
                p = {'t': ts-ts0, 'd': '<', 'l': len(buf), 'b': len(payload),'iat':ts-FLOWs[key_]["last_down_p"]}
                FLOWs[key_]["last_down_p"] = ts

                FLOWs[key_]["packets"].append(p)
                FLOWs[key_]["bytes_in"] += len(payload)
                FLOWs[key_]["pkts_in"] += 1

                FLOWs[key_]['downBD'] += get_BD(payload)
                if len(FLOWs[key_]['download']) < MAX_LOAD:
                    FLOWs[key_]['download'] += list(payload[0:MAX_LOAD - len(FLOWs[key_]['download'])])
    return FLOWs


if __name__ == '__main__':
    pass


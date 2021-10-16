#!/usr/bin/env python

from __future__ import division

import os
import sys
import subprocess
import signal
import socket
import logging
import thread
import time
import tempfile
import math
import random
import networkx as nx
import numpy as np

from k_shortest_paths import k_shortest_paths
from optparse import OptionParser
from bs4 import BeautifulSoup

# We need to import Python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Environment variable SUMO_HOME not defined")

import traci


class UnusedPortLock:
    lock = thread.allocate_lock()

    def __init__(self):
        self.acquired = False

    def __enter__(self):
        self.acquire()

    def __exit__(self):
        self.release()

    def acquire(self):
        if not self.acquired:
            UnusedPortLock.lock.acquire()
            self.acquired = True

    def release(self):
        if self.acquired:
            UnusedPortLock.lock.release()
            self.acquired = False


def find_unused_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    sock.bind(('127.0.0.1', 0))
    sock.listen(socket.SOMAXCONN)
    ipaddr, port = sock.getsockname()
    sock.close()

    return port


def terminate_sumo(sumo):
    if sumo.returncode == None:
        os.kill(sumo.pid, signal.SIGTERM)
        time.sleep(0.5)
        if sumo.returncode == None:
            os.kill(sumo.pid, signal.SIGKILL)
            time.sleep(1)
            if sumo.returncode == None:
                time.sleep(10)


def build_road_graph(network):
    # Input
    f = open(network)
    data = f.read()
    soup = BeautifulSoup(data, "xml")
    f.close()

    edges_length = {}

    for edge_tag in soup.findAll("edge"):
        lane_tag = edge_tag.find("lane")

        edge_id = edge_tag["id"]
        edge_length = float(lane_tag["length"])

        edges_length[edge_id] = edge_length

    graph = nx.DiGraph()

    for connection_tag in soup.findAll("connection"):
        source_edge = connection_tag["from"]
        dest_edge = connection_tag["to"]

        graph.add_edge(source_edge.encode("ascii"), dest_edge.encode("ascii"), length=edges_length[source_edge], weight=0)

    return graph


def log_densidade_speed(time):
    vehicles = traci.vehicle.getIDList()
    density = len(vehicles)
    speed = []

    output = open('output/average_speed_DSP.txt', 'a')

    for v in vehicles:
        lane_pos = traci.vehicle.getLanePosition(v)
        edge = traci.vehicle.getRoadID(v)
        if edge.startswith(":"):
            continue
        position = traci.vehicle.getPosition(v)
        route = traci.vehicle.getRoute(v)
        index = route.index(edge)
        if index > 0:
            distance = 500 * (index - 1) + lane_pos
        else:
            distance = lane_pos

        traveltime = traci.vehicle.getAdaptedTraveltime(v, time, edge)
        speed.append(float(distance)/float(time))

    if len(speed) > 0:
        output.write(str(np.amin(speed) * 3.6) + '\t' + str(np.average(speed)
                     * 3.6) + '\t' + str(np.amax(speed) * 3.6) + '\t' + str(density)+'\n')


def update_road_attributes(graph, time, begin_of_cycle, delta):
    # @dev - tentar calcular os valores dinamicamente

    # Valor default do Car-Following Model
    minGap = 2.5

    congestedRoads = set()

    for road in list(graph.nodes):
        travel_time = traci.edge.getAdaptedTraveltime(
            road.encode("ascii"), time)
        if travel_time <= 0:
            travel_time = traci.edge.getTraveltime(road.encode("ascii"))

        for successor_road in list(graph.successors(road)):
            Ki = traci.edge.getLastStepVehicleNumber(road.encode("ascii"))
            avgVehicleLength = traci.edge.getLastStepLength(road.encode("ascii"))
            Kjam = graph.edges[road, successor_road]["length"] / \
                (avgVehicleLength+minGap)
            if (Ki/Kjam) > delta:
                congestedRoads.add(road.encode("ascii"))

            # If it is the first measurement in a cycle, then do not compute the mean
            if begin_of_cycle:
                graph.edges[road, successor_road]["weight"] = travel_time
            else:
                t = (graph.edges[road, successor_road]
                    ["weight"] + travel_time) / 2
                t = t if t > 0 else travel_time
                graph.edges[road, successor_road]["weight"] = t

    return (graph, congestedRoads)


def dsp_reroute(vehicles, graph):
    for vehicle in vehicles:
        source = traci.vehicle.getRoadID(vehicle)
        if source.startswith(":"):
            continue
        route = traci.vehicle.getRoute(vehicle)
        destination = route[-1]

        if source != destination:
            logging.debug("Calculating shortest paths for pair (%s, %s)" % (source, destination))
            path = nx.dijkstra_path(graph, source, destination, weight='weight')
            
            traci.vehicle.setRoute(vehicle, path)


def select_vehicles(graph, congestedRoads, L):
    reverseGraph = nx.DiGraph.reverse(graph)
    selectedVehicles = set()

    for road in congestedRoads:
        count = 0
        bfs = []
        for edge in list(nx.bfs_edges(reverseGraph, road)):
            if edge[0] in bfs:
                count += 1
                bfs = []

            if count == L:
                break

            if count == 0:
                selectedVehicles = selectedVehicles.union(
                    set(traci.edge.getLastStepVehicleIDs(edge[0].encode('ascii'))))
            selectedVehicles = selectedVehicles.union(
                set(traci.edge.getLastStepVehicleIDs(edge[1].encode('ascii'))))

            bfs.append(edge[1])

    return selectedVehicles


def run(network, begin, end, interval, delta, urgency, level):
    logging.debug("Building road graph")
    road_graph_travel_time = build_road_graph(network)
    logging.debug("Finding all simple paths")

    # Used to enhance performance only
    buffered_paths = {}

    logging.debug("Running simulation now")
    step = 1
    # The time at which the first re-routing will happen
    # The time at which a cycle for collecting travel time measurements begins
    travel_time_cycle_begin = interval

    while step == 1 or traci.simulation.getMinExpectedNumber() > 0 or step != end:
        logging.debug("Minimum expected number of vehicles: %d" %
                      traci.simulation.getMinExpectedNumber())
        traci.simulationStep()
        log_densidade_speed(step)
        logging.debug("Simulation time %d" % step)

        if step >= travel_time_cycle_begin and travel_time_cycle_begin <= end:
            logging.debug(
                "Updating travel time on roads at simulation time %d" % step)
            road_graph_travel_time, congestedRoads = update_road_attributes(
                road_graph_travel_time, step, travel_time_cycle_begin == step, delta)

            selectedVehicles = select_vehicles(road_graph_travel_time, congestedRoads, level)
            if step % interval == 0 and len(congestedRoads) > 0:
                logging.debug("Rerouting vehicles at simulation time %d" % step)
                dsp_reroute(selectedVehicles, road_graph_travel_time)

        step += 1

    time.sleep(10)
    logging.debug("Simulation finished")
    traci.close()
    sys.stdout.flush()
    time.sleep(10)


def start_simulation(sumo, scenario, network, begin, end, interval, output, delta, urgency, level):
    logging.debug("Finding unused port")

    unused_port_lock = UnusedPortLock()
    unused_port_lock.__enter__()
    remote_port = find_unused_port()

    logging.debug("Port %d was found" % remote_port)

    logging.debug("Starting SUMO as a server")

    sumo = subprocess.Popen([sumo, "-c", scenario, "--tripinfo-output", output, "--device.emissions.probability",
                            "1.0", "--remote-port", str(remote_port)], stdout=sys.stdout, stderr=sys.stderr)
    unused_port_lock.release()

    try:
        traci.init(remote_port)
        run(network, begin, end, interval, delta, urgency, level)
    except Exception, e:
        logging.exception("Something bad happened")
    finally:
        logging.exception("Terminating SUMO")
        terminate_sumo(sumo)
        unused_port_lock.__exit__()


def main():
    # Option handling
    parser = OptionParser()
    parser.add_option("-c", "--command", dest="command", default="sumo",
                      help="The command used to run SUMO [default: %default]", metavar="COMMAND")
    parser.add_option("-s", "--scenario", dest="scenario", default="sim.sumocfg",
                      help="A SUMO configuration file [default: %default]", metavar="FILE")
    parser.add_option("-n", "--network", dest="network", default="sim.net.xml",
                      help="A SUMO network definition file [default: %default]", metavar="FILE")
    parser.add_option("-b", "--begin", dest="begin", type="int", default=1800, action="store",
                      help="The simulation time (s) at which the re-routing begins [default: %default]", metavar="BEGIN")
    parser.add_option("-e", "--end", dest="end", type="int", default=7200, action="store",
                      help="The simulation time (s) at which the re-routing ends [default: %default]", metavar="END")
    parser.add_option("-i", "--interval", dest="interval", type="int", default=600, action="store",
                      help="The interval (s) of classification [default: %default]", metavar="INTERVAL")
    parser.add_option("-o", "--output", dest="output", default="output/DSP-tripinfo.xml",
                      help="The XML file at which the output must be written [default: %default]", metavar="FILE")
    parser.add_option("--logfile", dest="logfile", default="log/sumo-launchd.log",
                      help="log messages to logfile [default: %default]", metavar="FILE")
    parser.add_option("-d", "--delta", dest="delta", type="float", default=0.7,
                      action="store", help="Congestion threshold [default: %default]", metavar="DELTA")
    parser.add_option("-l", "--level", dest="level", type="int", default=3, action="store",
                      help="Furthest distance a rerouted vehicle can be from congestion (in number of segments) [default: %default]", metavar="LEVEL")
    parser.add_option("-u", "--urgency", dest="urgency", default="ACI", action="store",
                      help="Urgency function used to compute the re-routing priority of a vehicle [default: %default]", metavar="URGENCY")

    (options, args) = parser.parse_args()

    logging.basicConfig(filename=options.logfile, level=logging.DEBUG)
    logging.debug("Logging to %s" % options.logfile)

    if args:
        logging.warning("Superfluous command line arguments: \"%s\"" % " ".join(args))

    start_simulation(options.command, options.scenario, options.network, options.begin, options.end,
                     options.interval, options.output, options.delta, options.urgency, options.level)

if __name__ == "__main__":
    main()

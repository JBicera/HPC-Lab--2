#!/usr/bin/env python3

# version 1.1 Hal Elrod Summer 2022 - addressed timeouts, error messages
# version 1.0 Hal Elrod Spring 2022
# portions copied from code by JR Smith
#
# Note: this is for performance testing _after_ code is working. It supresses error messages.

import csv
import re
import statistics
import subprocess
from math import ceil
from os import environ

NUM_ITERATIONS = 1  # Iterations of each collective/size to run (takes median if > 1)
TIMEOUT = 34  # Timeout (in seconds) running mpirun. This prevents runaways from hogging cluster
SCORE_MAX = 3  # Max points per collective/size test


def cmdline(command):
    """
    Runs environment (shell) command and looks for failure

    command : str - the shell command e.g. "mpirun"

	Raises: CalledProcessError (something unrecoverable with shell command happened)
    """
    try:
        process = subprocess.Popen(
            args=command,
            stdout=subprocess.PIPE,
            shell=True)
    except subprocess.CalledProcessError as e:
        print(f'Failure to execute command: {command}')
        raise e

    return process.communicate()[0]


def score(testtime, target, lower_threshold):
    """
    Scales the score to the points in the rubric

    testtime : float How long the collective took to run
    target : float - the 100% cutoff time
    lower_theshold : float - the minimum cutoff for getting points on the collective
    """
    if testtime > lower_threshold:  # don't reach min performance, no points for you
        return 0

    increment = (lower_threshold - target) / float(SCORE_MAX)  # scales the points
    iscore = min(SCORE_MAX - ceil((testtime - target) / increment),
                SCORE_MAX)  # you can't score more than max points - this time!
    return iscore


def runtest(testitem):
    """
    Run NUM_ITERATIONS tests on for the specific collective and size
    testitem : dictionary of parameters for each test
    """
    collective = testitem['collective']
    size = testitem['size']
    elements = testitem['param']
    targettime = float(testitem['target'])
    worsttime = float(testitem['minimum'])

    times = []
    for iteration in range(NUM_ITERATIONS):
        retry = True  # flag to retry any single iteration if it fails
        while retry:
            # Preferred for newer versions of MPI
            command = "mpirun --timeout {} ./{} -e {} 2>/dev/null".format(TIMEOUT, collective, elements)
            # Old way, for non-OpenMPI or versions < 3
            # command = 'timeout {} mpirun ./{} -e {} 2>/dev/null'.format(TIMEOUT, collective, elements)

            output = cmdline(command).decode('utf-8')
            match = re.search('[0-9].[0-9]+E-[0-9]+', output)
            if match:
                retry = False  # no need to retry
                time = float(match.group(0))
                # print(iteration, collective, size, f'{time:E}')
                times.append(time)
            else:
                if retry:  # if at first you don't succeed, try again. Once.
                    retry = False
                    continue
                else:
                    print(f'Failed execution and retry on {collective} {size} iteration {iteration}')
                    exit(1)

    try:
        median = statistics.median(times)
    except statistics.StatisticsError:
        print(f'Your collective did not run/return a valid time: {collective}')
        return 0

    weightscore = score(median, targettime, worsttime)
    print(collective, size, f'{median:.4E}', weightscore)
    return weightscore


def speedtest():
    """
    Speedtest - orchestrates the tests for all suppplied collectives, based on rubric file
    """
    cmdline("make all")
    cmdline("pwd")

    # run the tests!
    smallscore = largescore = 0
    smallcount = largecount = 0

    with open('rubric.csv', mode="r") as csv_file:  # "r" represents the read mode
        reader = csv.DictReader(csv_file)  # this is the reader object

        for item in reader:
            # you have to loop through the document to get each data
            if item['size'] == 'small':
                smallscore += runtest(item)
                smallcount += 1
            elif item['size'] == 'large':
                largescore += runtest(item)
                largecount += 1
            else:
                print('Something is wrong with rubric.csv file. Exiting')
                exit(1)

    # You can't get extra credit for exceeded total -- sorry
    smallscore = min(smallscore, smallcount * SCORE_MAX)
    largescore = min(largescore, largecount * SCORE_MAX)

    print(f'Small Score: {smallscore}')
    print(f'Large Score: {largescore}')

    return smallscore + largescore


if __name__ == '__main__':

    # Are you running on the root node? Then SLURM env isn't set. You should be ashamed!
    try:
        environ['SLURM_NPROCS']
    except KeyError:
        print('speedtest.py should be run in a PBS queue (either sbatch or pace_interactive -- DONT RUN ON ROOT NODE')

    # Did you forget to source ./pace_env? I've done that a hundred times. Here's a warning
    MPItype = cmdline('mpirun --version | grep "(Open MPI)"').decode('utf-8')
    if not re.search('Open MPI', MPItype):
        print('YOU ARE NOT RUNNING the correct version of MPI. Did you source ./pace_env.sh?')

score = speedtest()
exit(1)


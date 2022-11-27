# Python Implementation: unitTest.py
# -*- coding: utf-8 -*-
##
# @file       unitTest.py
#
# @version    1.1.1
#
# @par Purpose
# Unit test for GeneticAlgorithms package.
#
# @par Comments
# This programs performs a fully automated unit test of the class 
# or, at the users option, a graphic demo.GeneticAlgorithms
# This is Python 3 code!
#
# @par Known Bugs
# None
#
# @author     Ekkehard Blanz <Ekkehard.Blanz@gmail.com> (C) 2019-2022
#
# @copyright  See COPYING file that comes with this distribution
#

# File history:
#
#      Date         | Author         | Modification
#  -----------------+----------------+------------------------------------------
#   Tue Aug 20 2019 | Ekkehard Blanz | created
#   Sun Nov 06 2022 | Ekkehard Blanz | extracted fromm GeneticAlgorithms.py
#                   |                |
#


import sys
import matplotlib.pyplot as plt
import time
import unittest
import numpy as np
from GeneticAlgorithms import GeneticAlgorithms
from Genotype import Genotype

if "__main__" == __name__:


    class TestGenericAlgorithms( unittest.TestCase ):
        """!
        @brief Unit Test Class - all methods starting with test are
               automatically executed by unittest.
        """
        counter = 0

        @staticmethod
        def hookTest( ga ):
            TestGenericAlgorithms.counter += 1
            return

        @staticmethod
        def objfunc2chromosomes( *args ):
            return args[0][0] * args[0][1]

        def setUp( self ):
            return


        def tearDown( self ):
            return


        def testHaploidChromosomes( self ):
            ga = GeneticAlgorithms( objfunc1,
                                    1,
                                    32,
                                    30 )
            ga.run( 20 )
            genome, param, bestFit = ga.bestFit
            self.assertTrue( bestFit > 0.99 )
            self.assertTrue( len( ga.population ) == 30 )
            self.assertTrue( ga.generation == 20 )
            self.assertTrue( ga.pmx == False )
            del ga
            return


        def testEarlyExit( self ):
            ga = GeneticAlgorithms( objfunc1,
                                    1,
                                    32,
                                    30 )
            ga.run( 20, 0.5 )
            genome, param, bestFit = ga.bestFit
            self.assertTrue( bestFit >= 0.5 )
            self.assertTrue( len( ga.population ) == 30 )
            self.assertTrue( ga.pmx == False )
            del ga
            return


        def testDiploidChromosomes( self ):
            ga = GeneticAlgorithms( objfunc1,
                                    1,
                                    32,
                                    30,
                                    chromosomeSets=2 )
            ga.run( 20 )
            genome, param, bestFit = ga.bestFit
            self.assertTrue( bestFit > 0.99 )
            self.assertTrue( len( ga.population ) == 30 )
            self.assertTrue( ga.generation == 20 )
            self.assertTrue( ga.pmx == False )
            del ga
            return


        def testFloatChromosomes( self ):
            ga = GeneticAlgorithms( objfunc1,
                                    1,
                                    1,
                                    30,
                                    alleleAlphabet=float )
            ga.run( 40 )
            genome, param, bestFit = ga.bestFit
            self.assertTrue( bestFit > 0.99 )
            self.assertTrue( len( ga.population ) == 30 )
            self.assertTrue( ga.generation == 40 )
            self.assertTrue( ga.pmx == False )
            del ga
            return


        def testCharacterChromosomes( self ):
            pwd.pwd = "Hello"
            ga = GeneticAlgorithms(
                    pwd,
                    1,
                    len( pwd.pwd ),
                    10,
                    alleleAlphabet=GeneticAlgorithms.alnumAlphabet )
            ga.run( 800 )
            genome, param, bestFit = ga.bestFit
            self.assertTrue( bestFit > 0.99 )
            self.assertTrue( len( ga.population ) == 10 )
            self.assertTrue( ga.generation == 800 )
            self.assertTrue( ga.pmx == False )
            return


        def testBestNotImmortal( self ):
            ga = GeneticAlgorithms( objfunc1,
                                    1,
                                    32,
                                    20,
                                    bestImmortal=False )
            ga.run( 20 )
            genome, param, bestFit = ga.bestFit
            self.assertTrue( bestFit > 0.99 )
            self.assertTrue( len( ga.population ) ==20 )
            self.assertTrue( ga.generation == 20 )
            self.assertTrue( ga.pmx == False )
            del ga
            return


        def testPmx( self ):
            ga = GeneticAlgorithms( tsp,
                                    1,
                                    5,
                                    30,
                                    pmx=True )
            ga.run( 80 )
            genome, param, bestFit = ga.bestFit
            self.assertTrue( bestFit > 0.99 )
            self.assertTrue( len( ga.population ) == 30 )
            self.assertTrue( ga.generation == 80 )
            self.assertTrue( ga.pmx == True )
            del ga
            return


        def testMultiThreading( self ):
            ga = GeneticAlgorithms( objfunc1,
                                    1,
                                    32,
                                    30,
                                    threaded=True )
            ga.run( 20 )
            genome, param, bestFit = ga.bestFit
            self.assertTrue( bestFit > 0.99 )
            self.assertTrue( len( ga.population ) == 30 )
            self.assertTrue( ga.generation == 20 )
            self.assertTrue( ga.pmx == False )
            return


        def testMonogamous( self ):
            ga = GeneticAlgorithms( objfunc1,
                                    1,
                                    32,
                                    30,
                                    monogamous=True )
            ga.run( 20 )
            genome, param, bestFit = ga.bestFit
            self.assertTrue( bestFit > 0.99 )
            self.assertTrue( len( ga.population ) == 30 )
            self.assertTrue( ga.generation == 20 )
            self.assertTrue( ga.pmx == False )
            del ga
            return


        def testHook( self ):
            ga = GeneticAlgorithms( objfunc1,
                                    1,
                                    32,
                                    30,
                                    hook=self.hookTest )
            TestGenericAlgorithms.counter = 0
            ga.run( 20 )
            genome, param, bestFit = ga.bestFit
            self.assertTrue( bestFit > 0.99 )
            self.assertTrue( TestGenericAlgorithms.counter == 20 )
            self.assertTrue( len( ga.population ) == 30 )
            self.assertTrue( ga.generation == 20 )
            self.assertTrue( ga.pmx == False )
            del ga
            return


        def testPopulationGrowth( self ):
            ga = GeneticAlgorithms( objfunc1,
                                    1,
                                    32,
                                    20,
                                    populationGrowth=1.1 )
            ga.run( 20 )
            genome, param, bestFit = ga.bestFit
            self.assertTrue( bestFit > 0.99 )
            self.assertTrue( len( ga.population ) == 132 )
            self.assertTrue( ga.generation == 20 )
            self.assertTrue( ga.pmx == False )
            del ga
            return


        def testFourChildren( self ):
            ga = GeneticAlgorithms( objfunc1,
                                    1,
                                    32,
                                    30,
                                    numberChildren=4 )
            ga.run( 20 )
            genome, param, bestFit = ga.bestFit
            self.assertTrue( bestFit > 0.99 )
            self.assertTrue( len( ga.population ) == 30 )
            self.assertTrue( ga.generation == 20 )
            self.assertTrue( ga.pmx == False )
            del ga
            return


        def testErrorConditions1( self ):
            try:
                ga = GeneticAlgorithms( objfunc1,
                                        1,
                                        [32],
                                        30 )
            except ValueError:
                self.assertTrue( False )
            try:
                ga = GeneticAlgorithms( self.objfunc2chromosomes,
                                        2,
                                        32,
                                        30 )
            except ValueError as e:
                self.assertTrue( False )
            try:
                ga = GeneticAlgorithms( self.objfunc2chromosomes,
                                        2,
                                        [32],
                                        30 )
                self.assertTrue( False )
            except ValueError as e:
                pass
            try:
                ga = GeneticAlgorithms( objfunc1,
                                        1,
                                        [32,32],
                                        30 )
                self.assertTrue( False )
            except ValueError:
                pass
            return


        def testErrorConditions2( self ):
            try:
                ga = GeneticAlgorithms( tsp,
                                        1,
                                        5,
                                        30,
                                        pmx=True,
                                        alleleAlphabet=float )
                self.assertTrue( False )
            except ValueError:
                pass
            try:
                ga = GeneticAlgorithms( tsp,
                                        1,
                                        5,
                                        30,
                                        pmx=True,
                                        numberChildren=4 )
                self.assertTrue( False )
            except ValueError:
                pass
            try:
                ga = GeneticAlgorithms( tsp,
                                        1,
                                        5,
                                        30,
                                        pmx=True,
                                        chromosomeSets=2 )
                self.assertTrue( False )
            except ValueError:
                pass
            return


        def testErrorConditions3( self ):
            try:
                ga = GeneticAlgorithms( objfunc1,
                                        1,
                                        32,
                                        30,
                                        chromosomeSets=3 )
                self.assertTrue( False )
            except ValueError:
                pass
            return


        def testErrorConditions4( self ):
            try:
                ga = GeneticAlgorithms( objfunc1,
                                        1,
                                        32,
                                        30,
                                        alleleAlphabet=3 )
                self.assertTrue( False )
            except ValueError:
                pass
            try:
                ga = GeneticAlgorithms( objfunc1,
                                        1,
                                        32,
                                        30,
                                        alleleAlphabet=int )
                self.assertTrue( False )
            except ValueError:
                pass
            try:
                ga = GeneticAlgorithms( objfunc1,
                                        1,
                                        32,
                                        30,
                                        alleleAlphabet=float,
                                        chromosomeSets=2 )
                self.assertTrue( False )
            except ValueError:
                pass
            return


        def testErrorConditions5( self ):
            try:
                ga = GeneticAlgorithms( objfunc1,
                                        1,
                                        32,
                                        30,
                                        fitnessScale=0.9 )
                self.assertTrue( False )
            except ValueError:
                pass





    def printHelp():
        """!
        @brief Print command line help text.
        @return 0
        """
        helpText = \
"""
Synopsis:
    python3 unitTest.py <flag>
where flag can be -h or --help to print this help information, as well as -d or 
--demo to start the interactive demo program.  If no flag is given, the program 
executes its unit test.
"""
        print( helpText )
        return 0


    def objfunc0( *args ):
        """!
        @brief Objective function for demo from Goldberg's book (modified).

        This function takes on a single float argument in [0 .. 1].  The
        original function exhibits one maximum at the upper end of the interval,
        which would be trivially maximized by this class as it always includes
        the chromosomes with all ones (and that with all zeros) in the starting
        population.  This function is therefore slightly modified and exhibits
        the maximum at the center of the interval and a mirror image of itself
        in the upper half of the interval.
        @param *args args[0] argument of objective function in [0 .. 1]
        @return value of function [0 .. 1]
        """
        x = args[0]
        if x < 0.5:
            x *= 2
        else:
            x = 2 * (1 - x)
        return pow( x, 10 )


    def objfunc1( *args ):
        """!
        @brief Objective function 1 for demo.

        This function takes on a single float argument in [0 .. 1].  It exhibits
        two maxima in this interval, one at x=0.15 with objective value 1, the
        other at x=0.7 with objective value 0.82.  About 18.3% of the domain
        supports values above 0.82 and thus a population that can produce an
        individual with maximal fitness.
        @param *args args[0] argument of objective function in [0 .. 1]
        @return value of function [0 .. 1]
        """
        coeffs = [70.4499739424963,
                  -206.190728636476,
                  214.767969260518,
                  -95.9080356878612,
                  16.8808211213239,
                  0]
        x = args[0]
        power = 0
        retval = 0.
        for coeff in coeffs[::-1]:
            retval += coeff * x**power
            power += 1

        return max( 0., retval )


    def objfunc2( *args ):
        """!
        @brief Objective function 2 for demo.

        This function takes on a single float argument in [0 .. 1].  It exhibits
        three maxima in this interval, one at x=0.15 with objective value 1,
        the other two both very close to x=0.8 yielding a very broad second
        maximum with objective value around 0.85.  About 10.8% of the domain
        supports values above the second maximum and thus a population that can
        produce an individual with maximal fitness.
        @param *args args[0] argument of objective function in [0 .. 1]
        @return value of function [0 .. 1]
        """
        coeffs = [-1319.31857677856,
                  4529.00217356167,
                  -6016.32796026748,
                  3810.37520148061,
                  -1112.33243686956,
                  102.412465754810,
                  6.18913311850734,
                  0]
        x = args[0]
        power = 0
        retval = 0.
        for coeff in coeffs[::-1]:
            retval += coeff * x**power
            power += 1

        return max( 0., retval )


    def tsp( *args ):
        """!
        @brief Traveling Salesman Problem objective function.

        The function returns the inverse of the distance a salesman would
        travel given the provided tour of cities, given by their indices.  When
        the function is called for the first time (or when the parameter
        tsp.init is set to False), the function initializes a new set of city
        coordinates and inter-city distance matrix which is then kept constant
        through subsequent calls to tsp.  If the number of cities is less than
        12, the optimal path is computed and the returned value is then
        optDist / distance - otherwise, opDist is estimated and the result is
        then usually not bound by 1.  The coordinates of the cities are made
        available via the parameter tsp.coordinates.
        @param *args set of arguments (only the one element - array of
               coordinates)
        @return inverse of total distance (in [0 .. 1] for small number of
                cities)
        """

        answer = []

        def tspSolve( graph, v, currPos, n, count, cost ):
            """!
            @brief Function to find the minimum weight Hamiltonian Cycle
            """
            if (count == n and graph[currPos][0]):
                answer.append(cost + graph[currPos][0])
                return

            for i in range(n):
                if (v[i] == False and graph[currPos][i]):
                    v[i] = True
                    tspSolve( graph, v, i, n, count + 1,
                              cost + graph[currPos][i] )
                    v[i] = False
            return


        numberCities = len(  args[0] )

        try:
            if not tsp.init:
                raise AttributeError
        except AttributeError:
            # Setup TSP problem
            tsp.coordinates = np.random.random( (numberCities, 2) )
            tsp.distances = np.zeros( (numberCities, numberCities) )
            for i in range( numberCities ):
                for j in range( i + 1, numberCities ):
                    tsp.distances[i,j] = np.linalg.norm( tsp.coordinates[i,:] -
                                                         tsp.coordinates[j,:] )
                    tsp.distances[j,i] = tsp.distances[i,j]
            if numberCities < 12:
                v = [False for i in range( numberCities )]
                v[0] = True
                tspSolve( tsp.distances, v, 0, numberCities, 1, 0 )
                tsp.mindist = min( answer )
            else:
                print( "\nCannot compute optimal TSP solution - "
                       "problem too big - using estimate" )
                tsp.mindist = numberCities * 0.19
            tsp.init = True


        tour = list( copy.copy( args[0] ) )
        tour.append( tour[0] )
        distance = 0.
        previous = tour[0]
        for i in range( 1, numberCities + 1 ):
            distance += tsp.distances[previous,tour[i]]
            previous = tour[i]
        return tsp.mindist / distance


    def pwd( *args ):
        """!
        @brief Password guessing.
        @param args tuple with string with password guess
        @return percentage of correct characters at correct location [0 .. 1]
        """
        try:
            if type( pwd.pwd ) is not str:
                raise AttributeError
        except AttributeError:
            pwd.pwd = "Hello World!"
        guess = args[0]
        return sum( 1 for (p, g) in zip( pwd.pwd, guess) if p == g) / \
               len( pwd.pwd )


    def animate( ga ):
        """!
        @brief Plot animated progress of ga.

        For regular optimization cases, this function plots the objective
        function once and then the individuals of each generation as red dots on
        it; for TSP cases, it plots the cities and the identified best path for
        each generation.
        """
        try:
            if not animate.init:
                raise AttributeError
        except AttributeError:
            animate.lastPlot = time.time()
            animate.fig, animate.ax = plt.subplots( 2, 1 )
            if ga.pmx:
                # trick pyplot into setting the axes limits right
                x = [0, 1]
                y = [0, 1]
                animate.line = animate.ax[0].plot( x, y, "o-" )[0]
            else:
                x = np.linspace( 0, 1, 500 )
                y = [ga.objfunc( arg ) for arg in x]
                animate.ax[0].plot( x, y )
                animate.line = animate.ax[0].plot( [], [], "o" )[0]
            animate.ax[0].text( 0.78, 0.025, "Generation" )
            animate.label = animate.ax[0].text( 0.96, 0.025, "" )
            animate.init = True

        if ga.pmx:
            genome, param, bestFit = ga.bestFit
            x = list( tsp.coordinates[param[0],0] )
            x.append( x[0] )
            y = list( tsp.coordinates[param[0],1] )
            y.append( y[0] )
        else:
            x = []
            y = []
            for ind in ga.population:
                x.append( ga.decoder( ind ) )
                y.append( ind.fitness )
        animate.line.set_data( x, y )
        animate.label.set_text( str( ga.generation ) )
        now = time.time()
        deltaT = now - animate.lastPlot
        animate.lastPlot = now
        plt.pause( max( 0.001, 0.4 - deltaT ) )
        plt.draw()
        return



    def demo():
        """!
        @brief Interactive demo for GeneticAlgorithms.
        """

        # default values
        populationSize = 30
        numberChromosomes = 1
        chromosomeLengths = 32
        maxGenerations = 20
        chromosomeSets = 1
        pCrossover = 0.6
        pMutation = 0.0333
        pInversion = 0
        fitnessScale = 1.6
        populationGrowth = 1.0
        overpopulation = 1.3
        monogamous = False
        numberChildren = 2
        alleleAlphabet = [0,1]
        alleleAlphabetStr = "[0,1]"
        floatSigma = 0.05
        floatSigmaAdapt = 0.86
        bestImmortal = True
        threaded = False
        pmx = False
        hook = animate

        decoder = GeneticAlgorithms.genericDecoder

        print( "\n\nDefault values given in parentheses can be accepted by "
               "hitting 'enter'.\n" )
        print( "Currently available objective functions: "
               "objfunc0, objfunc1, objfunc2, tsp, and pwd" )
        print( "Currently available hook functions: "
               "animate, print, and None\n" )

        while True:
            objfunc = input( "Enter objective function ------------ > " )
            try:
                if objfunc and callable( eval( objfunc ) ):
                    objfunc = eval( objfunc )
                    break
            except NameError:
                pass
            print( "Error: {0} is not a valid objective function"
                   "".format( objfunc) )

        if objfunc == tsp:
            pmx = True
            alleleAlphabet = None
            alleleAlphabetStr = "None"
            chromosomeLengths = 10
            populationSize = 200
            maxGenerations = 100
            pCrossover = 0.9
            pMutation = 0.2
            fitnessScale = None
        elif objfunc == pwd:
            alleleAlphabet = GeneticAlgorithms.characterAlphabet
            alleleAlphabetStr = "characterAlphabet"
            pwd.pwd = input( "Enter password to guess ------------- > " )
            chromosomeLengths = len( pwd.pwd )
            populationSize = 10
            maxGenerations = 1000
            pCrossover = 0.2
            pMutation = 0.5
            fitnessScale = None
            hook = print

        while True:
            string = input( "Enter hook function ({0}) ------- > "
                            "".format( hook.__name__ ) )
            try:
                if not string:
                    break
                elif string == "None":
                    hook = None
                    break
                elif callable( eval( string ) ):
                    hook = eval( string )
                    break
            except NameError:
                pass
            print( "Error: {0} is not a valid hook function"
                   "".format( hook) )

        if not pmx:
            string = input( "Enter number of chromosome sets ({0}) - > "
                                "".format( chromosomeSets ) )
            if string:
                chromosomeSets = int( string )
                if chromosomeSets == 2:
                    alleleAlphabet = [-1,0,1]
                    alleleAlphabetStr = "[-1,0,1]"

        while not pmx:
            string = input( "Enter allele alphabet ({0}) ------- > "
                            "".format( alleleAlphabetStr ) )
            if string and (string == "float" or string == "real"):
                string = "float"
                chromosomeLengths = 1
                pCrossover = 0.0
                pMutation = 0.3
                maxGenerations = 100
                fitnessScale = None
                sstring = input( "Enter float sigma ({0:.2f}) ------------ > "
                                 "".format( floatSigma ) )
                if sstring:
                    floatSigma = float( sstring )
                sstring = input( "Enter float sigma adapt ({0:.2f}) ------ > "
                                 "".format( floatSigmaAdapt ) )
                if sstring:
                    floatSigmaAdapt = float( sstring )
            elif string and (string == "bool" or string == "binary"):
                string = "[0,1]"
            elif string and string == "alphaAlphabet":
                string = "GeneticAlgorithms.alphaAlphabet"
            elif string and string == "alnumAlphabet":
                string = "GeneticAlgorithms.alnumAlphabet"
            elif string and string == "characterAlphabet":
                string = "GeneticAlgorithms.characterAlphabet"
            if string:
                try:
                    alleleAlphabet = eval( string )
                    if type( alleleAlphabet ) is list or \
                       alleleAlphabet is float:
                        alleleAlphabetStr = string
                        break
                except:
                    pass
                print( "Error: valid alphabets are lists, or the types bool "
                       "and float" )
            else:
                break

        if not pmx:
            string = input( "Enter number of chromosomes ({0:4d}) -- > "
                            "".format( numberChromosomes ) )
            if string:
                numberChromosomes = int( string )
                chromosomeLengths = []
                for i in range( numberChromosomes ):
                    string = input( "Enter {0}. chromosome length ---------- > "
                                    "".format( i + 1 ) )
                    chromosomeLengths.append( int( string ) )
        if numberChromosomes == 1:
            string = input( "Enter chromosome length ({0:2d}) -------- > "
                            "".format( chromosomeLengths ) )
            if string:
                chromosomeLengths = int( string )

        string = input( "Enter population size ({0:4d}) -------- > "
                        "".format( populationSize ) )
        if string:
            populationSize = int( string )

        string = input( "Enter max. generations ({0:4d}) ------- > "
                        "".format( maxGenerations ) )
        if string:
            maxGenerations = int( string )
        string = input( "Enter crossover probability ({0:.2f}) -- > "
                        "".format( pCrossover ) )
        if string:
            pCrossover = float( string )
        string = input( "Enter mutation probability ({0:.2f}) --- > "
                        "".format( pMutation ) )
        if string:
            pCrossover = float( string )
        string = input( "Enter inversion probability ({0:.2f}) -- > "
                        "".format( pInversion ) )
        if string:
            pInversion = float( string )
        string = input( "Enter best immortal ({0}) ---------- > "
                        "".format( bestImmortal ) )
        if string:
            bestImmortal = bool( string )

        if fitnessScale is not None:
            string = input( "Enter fitness scale ({0:.2f}) ---------- > "
                            "".format( fitnessScale ) )
        else:
            string = input( "Enter fitness scale ({0}) ---------- > "
                            "".format( fitnessScale ) )
        if string:
            if string == "None" or string == "0":
                fitnessScale = None
            else:
                fitnesScale = float( string )
        string = input( "Enter population growth ({0:.2f}) ------ > "
                        "".format( populationGrowth ) )
        if string:
            populatinoGrowth = float( string )
        string = input( "Enter over-population ({0:.2f}) -------- > "
                        "".format( overpopulation ) )
        if string:
            overpopulation = float( string )
        string = input( "Enter monogamous ({0}) ------------ > "
                        "".format( monogamous ) )
        if not tsp:
            if string:
                monogamous = bool( string )
            string = input( "Enter number of children ({0:4d}) ----- > "
                            "".format( numberChildren ) )
        if string:
            numberChildren = int( string )
        string = input( "Enter threaded ({0}) -------------- > "
                        "".format( threaded ) )
        if string:
            threaded = bool( string )


        ga = GeneticAlgorithms( objfunc,
                                numberChromosomes,
                                chromosomeLengths,
                                populationSize,
                                decoder=decoder,
                                alleleAlphabet=alleleAlphabet,
                                pmx=pmx,
                                pCrossover=pCrossover,
                                pMutation=pMutation,
                                pInversion=pInversion,
                                populationGrowth=populationGrowth,
                                overpopulation=overpopulation,
                                chromosomeSets=chromosomeSets,
                                fitnessScale=fitnessScale,
                                monogamous=monogamous,
                                numberChildren=numberChildren,
                                bestImmortal=bestImmortal,
                                floatSigma=floatSigma,
                                floatSigmaAdapt=floatSigmaAdapt,
                                hook=hook,
                                threaded=threaded )

        ga.run( maxGenerations )

        print( "\nBest we could do after {0} generations..."
               "".format( maxGenerations ) )
        genome, param, bestFit = ga.bestFit
        print( "genome: {0}, parameter: {1} with fit: {2:.5f}"
               "".format( genome, param[0], bestFit ) )

        xs = np.linspace( 0, maxGenerations, maxGenerations + 1 )
        y1 = [s["max"] for s in ga.statistics]
        y2 = [s["mean"] for s in ga.statistics]

        if objfunc != pwd:
            if hook != animate:
                plt.subplot( 2, 1, 1 )
                x = np.linspace( 0, 1, 500 )
                if objfunc == tsp:
                    x = list( tsp.coordinates[param[0],0] )
                    x.append( x[0] )
                    y = list( tsp.coordinates[param[0],1] )
                    y.append( y[0] )
                    plt.plot( x, y, "o-" )
                else:
                    y = [objfunc( arg ) for arg in x]
                    plt.plot( x, y )
                    x = []
                    y = []
                    for ind in ga.population:
                        x.append( decoder( ind ) )
                        y.append( ind.fitness )
                    plt.plot( x, y, 'o' )
            plt.subplot( 2, 1, 2 )
            plt.plot( xs, y1, y2 )
            plt.legend( ["max","mean"], loc="lower right" )
            print( "\nType 'q' to quit" )
            plt.show()

        return 0


    def main():
        """!
        @brief Main Program - Unit Test and Demo

        Unit test for GeneticAlgorithms class and demo for its use and function.
        """

        try:
            if sys.argv[1] == "-h" or sys.argv[1] == "--help":
                return printHelp()
            elif sys.argv[1] == "-d" or sys.argv[1] == "--demo":
                return demo()
        except IndexError:
            pass

        unittest.main()


    sys.exit( int( main() or 0 ) )

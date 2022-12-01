# Python Implementation: Optimizer.
# -*- coding: utf-8 -*-
##
# @file       Optimizer.py
#
# @version    2.0.0
#
# @par Purpose
# Class to implement Genetic Algorithms using genes taking on values of settable
# alphabets (alleles) to form chromosomes which then are used to optimize
# parameters of an arbitrary objective function employing procedures from
# biologic evolution, i.e. "selective mating," "crossover," "mutation,"
# "selection of the fittest," and "inversion."
#
# @par Comments
# This is Python 3 code!
#
# @par Known Bugs
# None
#
# @author     Ekkehard Blanz <Ekkehard.Blanz@gmail.com> (C) 2019-2022
#
# @copyright  See COPYING file that comes with this distribution
#
# @mainpage
#
# @b Overview
# @par
#
# The class GOptimizer from this package implements a set of well-known
# Genetic Algorithms (GAs), in particular those propagated by Goldberg [1] for
# biallelic (binary) and triallelic (ternary) chromosomes and Schwefel [2] for
# those containing floating point numbers as the alphabet of their genes.  Those
# GAs can be used for parameter optimization for arbitrary objective functions.
# The "chromosomes" carrying the "genetic" information (i.e. the genotype
# leading to a phenotype which comprises the parameters to be optimized) are
# thought of containing genes with alleles of a given alphabet.  The alphabet
# is provided by the user and must be given either as a list of alleles or as
# the type float, in which case the elements of the alphabet are the floating
# point numbers in the interval [0 .. 1].  In another common case the alleles
# consist of the sequence of consecutive integers, and the genes are
# permutation of this sequence.  This option is selected by setting the pmx
# parameter (see below) in the constructor to True.  The user must also provide
# the objective function as a Python function and a Python decoder function
# that converts a given "genotype" (an instance of the class Genotype from this
# package) to the parameter list the objective function expects (the phenotype).
# One such generic decoder function is provided as a static method for biallelic
# haploid, triallelic diploid chromosomes, and those containing permutations of
# a sequence of integers as well as those with floating point alleles.  In this
# function, Biallelic chromosomes draw from the alphabet [0,1] and triallelic
# from the alphabet [-1,0,1], where -1 represents a recessive 1, 0 a 0, and 1 a
# dominant 1.  Alleles of type float and those consisting of sequences of
# integers are only supported for haploid chromosomes.  This generic mapping
# function works well for most simple cases where parameters should be kept
# separate in different chromosomes to protect them from inter-parameter
# crossover.   It is, of course, also possible to map the elements of an
# alphabet or floating point representation of multiple parameters onto a single
# chromosome.  The decoding of this arrangement, however, can no longer be
# implemented as a generic method since it requires task-specific knowledge
# about how many parameters there are, in the case of biallelic representation
# ([0,1] alphabet) how many bits should represent each parameter, and in which
# order they are; the decoding function must therefore be provided by the user
# in this case.  The user also specifies whether the chromosomes should be
# haploid or diploid, how many chromosomes there ought to be, and how long each
# of them is.  This class is oblivious to how the chromosomes are converted to
# the parameters for the objective function and what those parameters mean;
# all it knows internally are the chromosomes and the genes they contain.  Since
# the class does not use any numerical methods in the space of the parameters
# for the objective function, such as gradients, it is numerically very stable.
# The flip side of that is that GAs (just like their biological counterparts)
# are also rather "dumb" exactly because they cannot employ any information
# about the problem in any intelligent way - in a sense, they are the perfection
# of the application of dumb luck.
#
# This class then starts with a random population of genotypes with a given
# size that must be greater than the size of the allele alphabet (but usually
# should be much more than an order of magnitude bigger) and computes their
# "fitness" using the provided decoder to convert the genotype into the a
# phenotype and objective function, which serves as the environment.  The
# objective function must return a single "fitness" value which must be
# non-negative and monotonously rising with improved outcome quality or "better"
# outcome in whatever sense the user has in mind for the optimization.  This
# class then computes a new generation of a population using the genetic
# manipulations "selective mating," "crossover," "mutation," "selection of the
# fittest," and "inversion."  This process can be repeated arbitrarily many
# times.  In the case of biallelic chromosomes, it can be proven that the
# beneficial traits (schemata) represented in chromosomes grow exponentially
# from one generation to the next (Schema Theorem, see [1]).
#
# This class provides a rich set of parameters to fine-tune the behavior of
# the GAs.  Outside of the mentioned alphabet, which defaults to [0,1], and
# the type, number, and lengths of the chromosomes, there is first the size of
# the initial population.  Then there is a factor to control the population
# growth and one to control the amount of over-population before the selection
# of the fittest takes effect.  The former defaults to 1 (i.e. there is
# neither growth nor shrinking of the population from one generation to the
# next), and the latter defaults to 1.3 (i.e. the population will temporarily
# grow by a factor of 1.3 and the selection of the fittest will then kill off
# about one fourth of the population to get to the initial target population
# size).  Then, the probabilities for crossover and mutation can be selected.
# The former defaults to 0.6 for standard chromosomes, 0 for floating point
# chromosomes, and 0.9 for integer sequence chromosomes, the latter to 0.0333
# for standard chromosomes, to 0.3 for floating point chromosomes, and 0.4 for
#  integer sequence chromosomes.  The probability for crossover is on a
# per-chromosome and per offspring-pair base, that for mutation on a
# per-allele of all children case.  For integer sequence chromosomes, the
# crossover probability is interpreted as the probability with which two parent
# chromosomes exchange sequence elements in a procedure called partially matched
# crossover (pmx - see [1]).  The default probability for chromosome-sequence
# inversion is 0 in all cases.  Since the "selective mating" computes mating
# probabilities directly from the fitness values, dominant (but not optimal)
# individuals may lead to premature convergence and one may want to reduce the
# spread of the fitness values to encourage more diversity in that situation
# at the beginning.  Later, when most individuals cluster around the global
# optimum, one may want to increase the spread of the fitness values to avoid
# simply random selection of more or less equally performing individuals.  To
# enable an appropriate scaling of the fitness values, a parameter fitnessScale
# exist that can be set to some value between 1.2 and 2 to achieve this scaling,
# setting it to None switches fitness scaling off.  The default value for
# fitnessScale is 1.6 for standard chromosomes and None for integer sequence and
# real valued chromosomes.  Then, this class provides a parameter to select
# monogamous propagation, which, when set to True, will restrict each
# individual of any generation to produce offspring with one partner only; the
# default for this value is False.  In this context, it is also important to
# know that this class also allows to select the number of children each
# couple will produce, which defaults to 2 and must always be a multiple of 2.
# Since with this default setting the next generation will always have exactly
# the same number of individuals as the last when monogamous is set to True,
# couples will be allowed to divorce and re-marry until the target over
# population size is reached. Lastly, this class also allows to execute the
# evaluation of the objective function in as many threads as there are cores
# available. Obviously, this is particularly useful for computationally
# expensive objective functions.  In case of floating point chromosomes, there
# are also the variables floatSigma and floatSigmaAdapt to set the standard
# deviation of the normally distributed random numbers for mutation and
# its adaptation, respectively.  When the latter is set to 1, the standard
# deviation will not change based on the success rate of the mutations,
# otherwise it will be adjusted; the default value for the standard deviation
# is 1.2 and that for the standard deviation adaptation is 0.85 (see [2]).
#
# The class collects relevant fitness statistics from each generation and
# makes them available as a list of dictionaries.  The keys into these
# dictionaries are the strings: "mean", "variance", "min", "max",
# "crossovers", "mutations", "inversions", and "divorcerate," the latter only
# when monogamous is set to True.  The list is available via the attribute
# "statistics."
#
# Apart from these statistics, the class also provides the mutable properties
# pCrossover, pMutation, pInversion, populationGrowth, overPopulation, and
# fitnessScale, as well as floatSigma and floatSigmaAdapt to dynamically change
# these parameters, e.g. based on the collected statistics after any generation
# via a user-supplied function hook( ga ), which is given as its only argument
# the instance of this class that performs the genetic optimization.  Thereby,
# this class can be extended to almost arbitrary GAs that follow the basic
# principles of "selective mating," "crossover," "mutation," "crossover," and
# "selection of the fittest" using haploid or diploid sets of chromosomes with
# arbitrary allele alphabets.  Apart from the above mentioned read/write class
# properties, this class also provides the read-only properties objfunc,
# decoder, statistics, generation, population, pmx and best fit for the hook
# function to access the objective function, the decoding function, the
# statistics list, the number of the current generation, the current population,
# the pmx flag, and a tuple consisting of genome of best performing individual,
# phenotype of that individual as well as fitness value, respectively.
#
# @b Usage @b Notes
# @par
#
# To tackle a problem with this class, the first thing to do is think of an
# appropriate coding scheme to map the parameters to be optimized onto genes.
# The simplest way to do that is certainly to just use the parameters themselves
# as floating point values, which is easily accomplished by selecting the allele
# alphabet as type float.  If there are more than one parameter, the others can
# be arranged in the same chromosome or each as its own chromosome.  The latter
# method excludes the possibility of creating offspring using the crossover
# genetic procedure between the parameters in different chromosomes.
#
# It can be shown (see [1]) that smaller alphabets in general exhibit better
# performance than bigger ones, which renders the use of floating point numbers
# as the alphabet a less than optimal choice in most cases, as there are
# infinitely many alleles in the alphabet of real numbers in [0 .. 1].  It is
# therefore usually a better idea to use bits as individual genes and interpret
# them as bits of integers, which then can be converted to floating point
# numbers in [0 .. 1] in the decoding function if the application requires that.
# In that case it is usually a good idea to use separate chromosomes for
# different parameters to shield them from the effects of crossover between
# different parameters.
#
# Lastly, the method of partially matched crossover is provided for cases when
# the allele alphabet consists of integer sequences and the individual genomes
# in a population are permutations are permutations of each other.  A famous
# application for this type is the Traveling Salesman Problem (TSP) where the
# sequence of integers in each genome represents the sequence the salesman would
# visit the cities in on his tour.  However, to be more precise, what the GAs
# can solve in this case is the "blind" TSP, i.e. a case where the algorithm is
# unaware of the graph of cities or their distance matrix.  This is not a very
# interesting case mathematically, as the only hope for success is then dumb
# luck, which is, as we have seen, the area where GAs excel.  However, they do
# so only within a relatively narrow limit.  If the sequence length (the number
# of cities) becomes much larger than 20, GAs will perform poorly.
#
# The constructor of the class GOptimizer provides a number of sensible
# defaults for most of the parameters that should probably be tried first unless
# there are good reasons not to use them.
#
# The unit test included in this file can serve as an example of how to use this
# class.
#
# [1] David E. Goldberg, "Genetic Algorithms in Search, Optimization & Machine
# Learning," Addison-Wesley, 1989.
#
# [2] Hans-Paul Schwefel, "Numerische Optimierung von Computer-Modellen
# mittels Evolutionsstrategie (Numerical Optimization of Computer Models using
# Evolution Strategy)," Birkhäuser, 1977.
#

# File history:
#
#      Date         | Author         | Modification
#  -----------------+----------------+------------------------------------------
#   Tue Aug 20 2019 | Ekkehard Blanz | created
#   Mon Sep 16 2019 | Ekkehard Blanz | added first part of documentation
#   Mon Sep 16 2019 | Ekkehard Blanz | added character alphabets and separated
#                   |                | demo from unit test
#   Thu Oct 10 2019 | Ekkehard Blanz | improved unit test
#   Sun Nov 06 2022 | Ekkehard Blanz | extracted Genotype as separate module
#   Wed Nov 30 2022 | Ekkehard Blanz | renamed main class to Optimizer
#                   |                |


import multiprocessing
import copy
from threading import Thread
import numpy as np
from GeneticAlgorithms.Genotype import Genotype


class _ThreadWithReturnValue( Thread ):
    """!
    @brief Class derived from Thread that returns the return value of the
           threaded target function in the return value of join().
    """

    def __init__( self, group=None, target=None, name=None, Verbose=None,
                  args=(), kwargs={} ):
        """!
        @brief Constructor - calls parent's constructor and initializes
               return value.
        """
        Thread.__init__( self, group, target, name, args, kwargs )
        self._return = None
        return

    def run( self ):
        """!
        @brief Overridden run method to save return value.
        """
        if self._target is not None:
            self._return = self._target( *self._args, **self._kwargs )
        return

    def join( self, *args ):
        """!
        @brief Overridden join method returning return value.
        """
        Thread.join( self, *args )
        return self._return





class Optimizer( object ):
    """!
    @brief Genetic Algorithms (GA) optimizer for arbitrary objective functions.
    """

    ## alphabet (list) containing only ASCII letters and a blank
    alphaAlphabet = list(
        " " 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 
        "abcdefghijklmnopqrstuvwxyz" )
    ## alphabet (list) containing alphanumeric ASCII characters
    alnumAlphabet = copy.copy(
        alphaAlphabet )
    alnumAlphabet.extend( list ( "0123456789" ) )
    ## alphabet (list) containing characters on an American keyboard
    characterAlphabet = copy.copy(
        alnumAlphabet )
    characterAlphabet.extend( list(
        "~`!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?" ) )


    @staticmethod
    def genericDecoder( genotype ):
        """!
        @brief Generic decoder - static member function to map a biallelic or
               triallelic genotype into a phenotype consisting of a numpy array
               of either real values all in the unit interval or a (list of)
               strings.

        For numeric biallelic or triallelic chromosomes, all chromosomes in the
        genotype will be mapped into a single real value in [0 .. 1] each,
        treating each gene as a bit of an integer and then mapping the integer
        onto [0 .. 1].  In the case of floating point genes or if the partially
        matched crossover (pmx) flag is set, they are returned directly as an
        array, one array for each chromosome. If there was only one biallelic or
        triallelic chromosome, the return value will not be an array but rather
        a single real value in [0 .. 1] instead.  The same is true for a single
        valued chromosome.  For chromosomes whose alleles are characters, the
        return value is either a single string or a list of string if there was
        more than one chromosome.
        @param genotype instance of class Genotype
        @return tuple consisting of either a numpy array, a single real value,
                a string, or a list of strings
        """

        if genotype.alphabet is float or genotype.pmx:
            if genotype.diploid:
                raise ValueError( "Can only handle haploid chromosomes for "
                                  "float chromosomes and pmx" )
            genome = []
            i = 0
            for g in genotype.genome:
                genome.append( g.reshape( genotype.chromosomeLengths[i] ) )
                i += 1
            if len( genome ) > 1:
                return (genome,)
            elif len( genome[0] ) > 1:
                return (genome[0],)
            else:
                return (genome[0][0],)
        elif genotype.characterAlphabet:
            if genotype.diploid:
                raise ValueError( "Can only handle haploid chromosomes for "
                                  "character chromosomes" )
            if len( genotype.genome ) == 1:
                return ("".join( genotype.genome[0][:,0] ),)
            else:
                retval = []
                for chromosome in genotype.genome:
                    retval.append( "".join( chromosome[:,0] ) )
                return (retval,)
        else:
            if genotype.haploid:
                if genotype.alphabet.tolist() != [0, 1]:
                    raise ValueError( "Can only handle biallelic haploid "
                                      "chromosomes when alphabet is [0, 1]" )
                # here gene is a one-element vector - either 0 or 1
                express = lambda gene: gene[0]
            else:
                # we have diploid chromosomes
                if genotype.alphabet.tolist() != [-1, 0, 1]:
                    raise ValueError( "Can only handle triallelic diploid "
                                      "chromosomes when alphabet is "
                                      "[-1, 0, 1]" )
                # the alphabet [-1,0,1] has been chosen such that a simple max
                # function can find the expression considering dominant and
                # recessive alleles
                express = lambda gene: abs( max( gene ) )
            retval = []
            for chromosome in genotype.genome:
                accu = 0.0
                power2 = 2**(len( chromosome ) - 1)

                for gene in chromosome:
                    accu += express( gene ) * power2
                    power2 //= 2
                retval.append( accu / (2**len( chromosome ) - 1) )

            if len( genotype.genome ) == 1:
                return (retval[0],)
            else:
                return (np.array( retval ),)


    def __init__( self,
                  objectiveFunction,
                  numberChromosomes,
                  chromosomeLengths,
                  populationSize,
                  decoder=None,
                  alleleAlphabet=None,
                  pmx=False,
                  pCrossover=None,
                  pMutation=None,
                  pInversion=None,
                  populationGrowth=1.0,
                  overpopulation=1.3,
                  chromosomeSets=1,
                  fitnessScale=0.,
                  monogamous=False,
                  numberChildren=2,
                  bestImmortal=True,
                  floatSigma=1.2,
                  floatSigmaAdapt=0.85,
                  hook=None,
                  threaded=False ):
        """!
        @brief Constructor - check validity of parameters and set up first
               random population.

        Default values are given in parameters.
        @param objectiveFunction user-supplied Python function
        @param numberChromosomes number of chromosomes
        @param chromosomeLengths list of lengths of chromosomes - can be int
               if all chromosomes have the same length
        @param populationSize size of initial population
        @param decoder user-suppliedPython function (genericDecoder)
        @param alleleAlphabet alphabet list or type float from which to draw
               alleles ([0, 1] for haploid and [-1,0,1] for diploid chromosomes)
        @param pmx set True to enable partially matched crossover (False)
        @param pCrossover crossover probability (depending on method)
        @param pMutation mutation probability (depending on method)
        @param pInversion inversion probability
        @param populationGrowth growth factor between populations (1.0)
        @param overpopulation population overshoot before selection (1.2)
        @param chromosomeSets number of complete chromosome sets (1 - haploid)
        @param fitnessScale factor for fitness scaling (1.6 for regular
               chromosomes and None for float and pmx)
        @param monogamous monogamy parameter (False)
        @param numberChildren number of children per mating (2)
        @param bestImmortal set False to not have the best individual survive
               unchanged to the next generation (True)
        @param floatSigma standard deviation for float-type chromosome mutation
               (1.2)
        @param floatSigmaAdapt standard deviation adaptation for random
               mutation (0.85)
        @param hook user-supplied hook function called after every generation
               (None)
        @param threaded use threads when set to True (False)
        """
        self.__objf = objectiveFunction
        self.__decf = decoder
        if decoder is None:
            self.__decf = self.genericDecoder
        self.__populationSize = populationSize
        self.__bestImmortal = bestImmortal

        if type( chromosomeLengths ) is not list:
            chromosomeLengths = [chromosomeLengths] * numberChromosomes
        if numberChromosomes != len( chromosomeLengths ):
            raise ValueError( "chromosomeLengths must have exactly"
                              "numberChromosomes elements" )

        self.__numberChromosomes = numberChromosomes
        self.__chromosomeLengths = chromosomeLengths

        if pmx:
            if alleleAlphabet is float:
                raise ValueError( "For PMX, the alphabet cannot be float" )
            if alleleAlphabet is None:
                alleleAlphabet = np.arange( chromosomeLengths[0] )
            if any( l != len( alleleAlphabet ) for l in chromosomeLengths ):
                raise ValueError( "For PMX, the chromosome length must be "
                                  "the same as the alphabet length "
                                  "in all chromosomes" )
            if chromosomeSets != 1:
                raise ValueError( "PMX only works with haploid chromosome "
                                  "sets" )
            if numberChildren != 2:
                raise ValueError( "PMX always produces exactly two children" )
        self.__pmx = pmx

        if chromosomeSets != 1 and chromosomeSets != 2:
            raise ValueError( "The number of chromosome sets can only be 1 or "
                              "2" )
        self.__chromosomeSets = chromosomeSets

        if alleleAlphabet is None:
            if self.__chromosomeSets == 1:
                self.__alleleAlphabet = np.array( [0, 1] )
            else:
                self.__alleleAlphabet = np.array( [-1, 0, 1] )
        elif type( alleleAlphabet ) is list:
            dtype = type( alleleAlphabet[0] )
            # prevent numpy's "intelligent" type unification if not all
            # elements of the alphabet are of the same type; this allows
            # alphabets of the form [0, 1, '*'] where 0 and 1 can still be
            # treated as integers and '*' can raise an exception
            if any( not isinstance( a, dtype ) for a in alleleAlphabet[1:] ):
                dtype = object
            self.__alleleAlphabet = np.array( alleleAlphabet, dtype=dtype )
        elif alleleAlphabet is float or type( alleleAlphabet ) is np.ndarray:
            self.__alleleAlphabet = alleleAlphabet
        else:
            raise ValueError( "The allele alphabet must be a list, an ndarray, "
                              "or the type float" )

        if self.__alleleAlphabet is float and self.__chromosomeSets == 2:
            raise ValueError( "Allele alphabets of type float only work with "
                              "haploid chromosomes" )
        try:
            self.__characterAlphabet = \
                all( a.isprintable() for a in self.__alleleAlphabet )
        except (AttributeError, TypeError):
            self.__characterAlphabet = False

        # the following eight properties don't need setters or getters
        # they can be changed between successive calls to run() - default values
        # for many of them depend on the data type of the genes

        ## The probability of gene-crossover on a per chromosome gene basis.
        self.pCrossover = pCrossover
        ## The probability for a gene to mutate on a per gene basis.
        self.pMutation = pMutation
        ## The probability for a sequence of genes to be inverted.
        self.pInversion = pInversion
        ## Factor of net population growth per generation.
        self.populationGrowth = populationGrowth
        ## Over-population factor before selection of the fittest happens.
        self.overpopulation = overpopulation
        ## Artificial spread of the fitness function (1.2 to 2.0)
        self.fitnessScale = fitnessScale
        ## Sigma for random numbers varying floating point genes
        self.floatSigma = floatSigma
        ## Modification factor for sigma floatSigma
        self.floatSigmaAdapt = floatSigmaAdapt

        if pCrossover is None:
            if self.__alleleAlphabet is float:
                self.pCrossover = 0.0
            elif self.__pmx:
                self.pCrossover = 0.9
            else:
                self.pCrossover = 0.6
        else:
            self.pCrossover = pCrossover

        if pMutation is None:
            if self.__alleleAlphabet is float:
                self.pMutation = 0.3
            elif self.__pmx:
                self.pMutation = 0.4
            else:
                self.pMutation = 0.0333
        else:
            self.pMutation = pMutation

        if pInversion is None:
            if self.__alleleAlphabet is float:
                self.pInversion = 0.0
            elif self.__pmx:
                self.pInversion = 0.0
            else:
                self.pInversion = 0.0
        else:
            self.pInversion = pInversion

        if fitnessScale == 0.:
            # take care of default values in this case when 0.
            if self.__alleleAlphabet is float or self.__characterAlphabet \
               or self.__pmx:
                self.fitnessScale = None
            else:
                self.fitnessScale = 1.6
        else:
            self.fitnessScale = fitnessScale
        if self.fitnessScale is not None and self.fitnessScale < 1.0:
            raise ValueError( "Fitness scale must be > 1 or None" )

        self.__monogamous = monogamous
        if numberChildren % 2 != 0:
            raise ValueError( "Number of children must be a multiple of 2" )
        self.__numberChildren = numberChildren

        self.__threaded = threaded

        self.__statistics = []
        self.__population = []

        for __ in range( populationSize ):
            self.__population.append( Genotype( numberChromosomes,
                                                chromosomeLengths,
                                                self.__chromosomeSets,
                                                self.__alleleAlphabet,
                                                pmx=self.__pmx ) )

        if self.__alleleAlphabet is not float and not self.__pmx and \
           not self.__characterAlphabet and \
           populationSize > len( self.__alleleAlphabet ):
            # make sure all genes have all possible values of the allele
            # alphabet represented in the initial population
            for k in range( len( self.__alleleAlphabet ) ):
                fringeGenome = []
                for i in range( numberChromosomes ):
                    fringeGenome.append( np.full( (chromosomeLengths[i],
                                                   self.__chromosomeSets),
                                                  self.__alleleAlphabet[k] ) )
                self.__population[k] = Genotype( numberChromosomes,
                                                 chromosomeLengths,
                                                 self.__chromosomeSets,
                                                 self.__alleleAlphabet,
                                                 fringeGenome )

        self.__hook = hook

        self.__evaluate( self.__population )

        # set statistics to zero and append them
        self.__crossovers = 0
        self.__mutations = 0
        self.__inversions = 0
        self.__divorceRate = 0.0

        self.__appendStatistics( self.__population )

        if self.__hook is not None:
            self.__hook( self )

        return


    def __str__( self ):
        """!
        @brief  Assemble and return string with current internal parameters and
                current state of the population.
        """
        retstr = "\n"
        retstr += "Problem-specific parameters:\n"
        retstr += "numberChromosomes: {0}, chromosomeLengths: {1}" \
                  "\n".format( self.__population[0].numberChromosomes,
                               self.__population[0].chromosomeLengths )
        retstr += "\nGA-specific parameters:\n"
        retstr += "populationSize: {0}, populationGrowth: {1}, " \
                  "overpopulation: {2}\n" \
                  "chromosomeSets: {3}, " \
                  "pCrossover: {4}, pMutation: {5}, pInversion: {6}, " \
                  "fitnessScale: {7}\n" \
                  "monogamous: {8}, numberChildren: {9}\nthreaded: {10}" \
                  "\n".format( self.__populationSize,
                               self.populationGrowth,
                               self.overpopulation,
                               self.__chromosomeSets,
                               self.pCrossover,
                               self.pMutation,
                               self.pInversion,
                               self.fitnessScale,
                               self.__monogamous,
                               self.__numberChildren,
                               self.__threaded )
        if self.__alleleAlphabet is float:
            retstr += "\nParameters for floating point chromosomes:\n"
            retstr += "floatSigma: {0}, floatSigmaAdapt: {1}\n" \
                      "".format( self.floatSigma, self.floatSigmaAdapt )
        retstr += "\nNumber of generations computed: "\
                  "{0}\n".format( self.generation )
        retstr += "\nCurrent Population:\n"
        for individual in self.__population:
            retstr += str( individual )
            retstr += "; "
        retstr = retstr[:-2] + "\n"
        if self.__alleleAlphabet is not float and not self.__pmx:
            retstr += "Decoded:\n"
            for individual in self.__population:
                retstr += str( self.__decf( individual ) )
                retstr += "; "
        retstr += "\nFitness:\n"
        for individual in self.__population:
            retstr += str( individual.fitness )
            retstr += "; "
        retstr = retstr[:-2] + "\n"
        retstr += "\nCurrent Statistics:\n{0}\n".format( self.__statistics[-1] )
        return retstr


    def run( self, generations, maxfit=None ):
        """!
        @brief Run a defined number of generations or until maximum fitness is
               reached.
        @param generations number of generations to produce
        @param maxfit abortion criterion when fitness reached maximum (default
               is None)
        """
        for __ in range( generations ):
            self.__nextGen()
            if maxfit is not None and self.bestFit[2] >= maxfit:
                break
            if self.__hook is not None:
                self.__hook( self )
        return


    def __evaluate( self, population ):
        """!
        @brief Evaluate given population and set fitness attribute of elements.
        @param population [in, out] (mutable) any list of genotypes
        """
        if self.__threaded:
            cores = min( multiprocessing.cpu_count(), len( population ) )
            threads = [None] * cores
            indices = [None] * cores
            threadNo = 0

            for i, individual in enumerate( population ):
                threadNo = i % cores
                if i >= cores:
                    population[indices[threadNo]].fitness = \
                                                        threads[threadNo].join()
                    population[indices[threadNo]].scaledFitness = \
                                           population[indices[threadNo]].fitness
                threads[threadNo] = \
                    _ThreadWithReturnValue( target=self.__objf,
                                            args=self.__decf( individual ) )
                threads[threadNo].daemon = True
                threads[threadNo].start()
                indices[threadNo] = i

            threadNo += 1
            for i in range( threadNo, threadNo + cores ):
                threadNo = i % cores
                population[indices[threadNo]].fitness = threads[threadNo].join()
                population[indices[threadNo]].scaledFitness = \
                                           population[indices[threadNo]].fitness
        else:
            for individual in population:
                args = self.__decf( individual )
                individual.fitness = self.__objf( *args )
                individual.scaledFitness = individual.fitness
        return


    def __scaleFitness( self, population ):
        """!
        @brief Scale fitness of given population.

        This scaling assures that the new fitness is always non-negative.  While
        observing that restriction, it tries to scale the maximum fitness up to
        self.fitnessScale * favg, or else as much as it can to still obey the
        restriction.
        NOTE: It is important that appendStatistics() has been called on the
        same population before calling this method.
        @param population [in, out] (mutable) any list of genotypes
        """
        if self.fitnessScale is None:
            # identical scaledFitness was already entered by __evaluate()
            return

        fmin = self.__statistics[-1]["min"]
        favg = self.__statistics[-1]["mean"]
        fmax = self.__statistics[-1]["max"]
        if favg <= (fmax + fmin * (self.fitnessScale - 1)) / self.fitnessScale:
            a = favg * (self.fitnessScale - 1) / (fmax - favg)
        else:
            a = favg / (favg - fmin)
        b = favg * (1 - a)
        for individual in population:
            # also make sure we don't run into round-off errors
            individual.scaledFitness = max( 0, individual.fitness * a + b )
        return


    def __maxFit( self ):
        """!
        @brief Find fitness of best performing individual in current population.
        """
        mf = self.__population[0].fitness
        for individual in self.__population:
            if individual.fitness > mf:
                mf = individual.fitness
        return mf


    def __nextGen( self ):
        """!
        @brief Compute the next generation, the fitness of its individuals, and
               its statistics.

        A new population is created as a list of individual genotypes, and their
        fitness is evaluated.  Then natural selection is performed and the
        statistics of the remaining new population are recorded.
        """
        newpop = []
        consumed = []
        self.__crossovers = 0
        self.__mutations = 0
        self.__inversions = 0
        if self.__monogamous:
            self.__divorceRate = 0.0
        else:
            self.__divorceRate = 1.0

        targetSize = int( self.overpopulation * self.__populationSize * \
                          self.populationGrowth )
        if self.__bestImmortal:
            targetSize -= 1

        while len( newpop ) < targetSize:

            # Selective Mating - find partners i and j
            i = self.__findMate( consumed )
            consumed.append( i )
            if len( consumed ) >= len( self.__population ):
                # if we run out of partners, we divorce everybody
                consumed = []
            j = self.__findMate( consumed )

            # Generation of offspring
            if self.__pmx:
                # perform partially matched crossover
                children = self.__swapSections( self.__population[i],
                                                self.__population[j] )
            elif self.__chromosomeSets == 1:
                # perform simple crossover
                children = self.__crossover( self.__population[i],
                                             self.__population[j] )
            else:
                # perform meiosis
                hapchromsI = self.__makeHaploid( self.__population[i] )
                hapchromsJ = self.__makeHaploid( self.__population[j] )
                gametesI = self.__crossover( hapchromsI[0], hapchromsI[1] )
                gametesJ = self.__crossover( hapchromsJ[0], hapchromsJ[1] )
                children = self.__fertilization( gametesI, gametesJ )

            # Mutation of children generated above
            for child in children:
                self.__mutation( child )
                self.__inversion( child )

            if self.__monogamous:
                consumed.append( j )
            else:
                consumed = []

            newpop.extend( children )
            if len( consumed ) >= len( self.__population ):
                # if we run out of partners, we divorce everybody
                self.__divorceRate = \
                    (len( self.__population ) - len( consumed )) / \
                    len( self.__population )
                consumed = []

        if self.__bestImmortal:
            # add best individual from last generation not to loose it
            lastBest, dummy1, dummy2 = self.bestFit
            newpop.append( lastBest )

        self.__evaluate( newpop )

        self.__appendStatistics( newpop )

        self.__scaleFitness( newpop )

        # Selection of the Fittest
        self.__population = self.__selection( newpop )

        if self.__alleleAlphabet is float:
            if self.generation > 0 and self.generation % 5 == 0:
                if self.__statistics[-1]["mean"] > \
                   self.__statistics[-5]["mean"]:
                    self.floatSigma *= self.floatSigmaAdapt
                else:
                    self.floatSigma /= self.floatSigmaAdapt


        return


    def __findMate( self, consumed ):
        """!
        @brief Find index for a mate other than those in list consumed.

        The selection is biased by the fitness of the individuals in the
        population such that fitter individuals have a higher probability of
        being selected.
        @param consumed list of individuals excluded from being selected
        @return index of selected individual in self.__population
        """
        assert( len( consumed ) <= self.__populationSize )
        fitnessSum = 0.
        for i, individual in enumerate( self.__population ):
            if i not in consumed:
                fitnessSum += individual.scaledFitness
        limit = np.random.random() * fitnessSum
        partsum = 0
        for j in range( self.__populationSize ):
            if j in consumed: continue
            partsum += self.__population[j].scaledFitness
            if partsum >= limit: break
        while j >= self.__populationSize:
            # no j found -
            # find selectable element with highest index regardless of fitness
            j -= 1
            if j not in consumed:
                break
        return min( j, self.__populationSize - 1 )


    def __crossover( self, genome1, genome2 ):
        """!
        @brief Generate new haploid genomes from two existing haploid genomes by
               performing a simple crossover between individual genomes.

        The probability pCrossover is applied for each chromosome for each pair
        of results.

        This method is used for haploid chromosomes to produce children directly
        and for diploid chromosomes to produce gametes.
        @param genome1 haploid instance of Genotype of first genome
        @param genome2 haploid instance of Genotype of second genome
        @return list of haploid instances of Genotypes of all results
        """

        assert( genome1.haploid and genome2.haploid )

        # prepare empty (haploid) results genomes
        results = []
        for i in range( self.__numberChildren ):
            results.append( [] )
            for j in range( self.__numberChromosomes ):
                results[i].append(
                    np.empty( shape=(self.__chromosomeLengths[j],1),
                              dtype=genome1.genome[0].dtype ) )

        for n in range( 0, self.__numberChildren, 2 ):
            # each mating process produces 2 results

            jcross = [] # one crossover point for each chromosome
            for i in range( self.__numberChromosomes ):
                if np.random.random() <= self.pCrossover:
                    jcross.append(
                        np.random.randint( 1, self.__chromosomeLengths[i] ) )
                    self.__crossovers += 1
                else:
                    jcross.append( self.__chromosomeLengths[i] )

            for i in range( self.__numberChromosomes ):
                for j in range( jcross[i] ):
                    results[n][i][j][0] = genome1.genome[i][j][0]
                    results[n+1][i][j][0] = genome2.genome[i][j][0]
                for j in range( jcross[i], self.__chromosomeLengths[i] ):
                    results[n][i][j][0] = genome2.genome[i][j][0]
                    results[n+1][i][j][0] = genome1.genome[i][j][0]

        # now convert genomes of results into proper genotypes
        retval = []
        for child in results:
            retval.append( Genotype( self.__numberChromosomes,
                                     self.__chromosomeLengths,
                                     1,
                                     self.__alleleAlphabet,
                                     child ) )
        return retval


    def __mutation( self, genotype ):
        """!
        @brief Mutate the genes of a given genotype.

        The probability pMutation is applied on a per gene basis.  If the
        alphabet is discrete and its length greater than 2, the gene to which
        the mutation happens is random and equally distributed among all other
        elements of the alphabet; if the alphabet length is 2, it is always the
        other allele.  If the alphabet is float, the mutation is a normally
        distributed distortion of the value with variance self.floatSigma**2.
        In the case of partially matched crossover, only permutations of the
        complete alphabet are allowed, hence a mutation is implemented as an
        exchange of the alleles at two random locations.  Sine this affects
        two locations, the mutation probability is halved in that case and each
        exchange is counted as two mutations.
        @param genotype [in, out] (mutable) genotype to mutate
        """

        if self.pMutation == 0:
            return

        threshold = self.pMutation
        increment = 1
        if self.__pmx:
            threshold /= 2.
            increment = 2

        for i in range( genotype.numberChromosomes ):
            for j in range( genotype.chromosomeLengths[i] ):
                for k in range( self.__chromosomeSets ):
                    if np.random.random() <= threshold:
                        if self.__pmx:
                            jAlt = np.random.randint(
                                                 genotype.chromosomeLengths[i] )
                            genotype.genome[i][j][k], \
                            genotype.genome[i][jAlt][k] = \
                                genotype.genome[i][jAlt][k], \
                                genotype.genome[i][j][k]
                        elif self.__alleleAlphabet is float:
                            genotype.genome[i][j] = \
                                min( 1.,
                                    max( 0.,
                                        genotype.genome[i][j] +
                                        np.random.normal( 0.,
                                                          self.floatSigma ) ) )
                        elif len( self.__alleleAlphabet ) == 2:
                            # since binary genes are often used, we provide a
                            # method that works without a call to
                            # np.random.randint
                            r = np.where( self.__alleleAlphabet ==
                                          genotype.genome[i][j][k] )[0]
                            genotype.genome[i][j][k] = \
                                self.__alleleAlphabet[1 - r]
                        else:
                            rExclude = np.where( self.__alleleAlphabet ==
                                                 genotype.genome[i][j][k] )[0]
                            r = np.random.randint( len( self.__alleleAlphabet )
                                                   - 1 )
                            if r >= rExclude:
                                r += 1
                            genotype.genome[i][j][k] = self.__alleleAlphabet[r]
                        self.__mutations += increment
        return


    def __inversion( self, genotype ):
        """!
        @brief Invert a sequence of genes of a given genotype.

        The probability pInversion is applied on a per chromosome basis.
        @param genotype [in, out] (mutable) genotype to mutate
        """

        if self.pInversion == 0:
            return

        for i in range( genotype.numberChromosomes ):
            for k in range( self.__chromosomeSets ):
                if np.random.random() <= self.pInversion:
                    j1 = np.random.randint( genotype.chromosomeLengths[i] )
                    j2 = np.random.randint( genotype.chromosomeLengths[i] )
                    if j1 > j2:
                        j1, j2 = j2, j1
                    jAlt = j2
                    for j in range( j1, j2 ):
                        genotype.genome[i][j][k], \
                        genotype.genome[i][jAlt][k] = \
                            genotype.genome[i][jAlt][k], \
                            genotype.genome[i][j][k]
                        jAlt -= 1
                    self.__inversions += 1
        return


    def __makeHaploid( self, genotype ):
        """!
        @brief Make two haploid genotypes out of one diploid genotype.
        @param genotype diploid instance of class Genotype to split
        @return tuple of two haploid instances of class Genotype
        """
        assert( genotype.haploid == False )

        hap1 = []
        hap2 = []
        for i in range( self.__numberChromosomes ):
            hap1.append( np.reshape( np.array( genotype.genome[i][:,0] ),
                                     (genotype.chromosomeLengths[i], 1) ) )
            hap2.append( np.reshape( np.array( genotype.genome[i][:,1] ),
                                     (genotype.chromosomeLengths[i], 1) ) )
        return (Genotype( self.__numberChromosomes,
                          self.__chromosomeLengths,
                          1,
                          self.__alleleAlphabet,
                          hap1 ),
                Genotype( self.__numberChromosomes,
                          self.__chromosomeLengths,
                          1,
                          self.__alleleAlphabet,
                          hap2 ))


    def __fertilization( self, gametesMat, gametesPat ):
        """!
        @brief Fertilize (haploid) maternal gametes with (haploid) paternal
               gametes one-to-one to create as many diploid offspring as there
               were gametes.
        @param gametesMat list of haploid maternal instances of Genotype
        @param gametesPat list of haploid paternal instances of Genotype
        @return list of genotypes of self.__numberChildren children
        """
        assert( len( gametesPat ) == len( gametesMat ) )
        assert( len( gametesPat ) == self.__numberChildren )

        children = []
        for i in range( self.__numberChildren ):
            genome = []
            for j in range( self.__numberChromosomes ):
                chromosome = np.empty( (self.__chromosomeLengths[j], 2),
                                       dtype=gametesPat[i].genome[j].dtype )
                chromosome[:,0] = gametesMat[i].genome[j][:,0]
                chromosome[:,1] = gametesPat[i].genome[j][:,0]
                genome.append( chromosome )
            children.append( Genotype( self.__numberChromosomes,
                                       self.__chromosomeLengths,
                                       self.__chromosomeSets,
                                       self.__alleleAlphabet,
                                       genome ) )

        return children


    def __swapSections( self, genome1, genome2 ):
        """!
        @brief Perform "partially matched crossover" by swapping the alleles
               of the parents in a randomly picked region and adjusting the
               rest of  the alleles such that each child is again a valid
               permutation of the alphabet.
        @param genome1 instance of maternal Genotype
        @param genome2 instance of paternal Genotype
        @return list of genotypes of self.__numberChildren children
        """
        children = []
        children.append( Genotype( self.__numberChromosomes,
                                   self.__chromosomeLengths,
                                   self.__chromosomeSets,
                                   self.__alleleAlphabet,
                                   genome1.genome,
                                   self.__pmx ) )
        children.append( Genotype( self.__numberChromosomes,
                                   self.__chromosomeLengths,
                                   self.__chromosomeSets,
                                   self.__alleleAlphabet,
                                   genome2.genome,
                                   self.__pmx ) )
        for i in range( children[0].numberChromosomes ):
            if np.random.random() > self.pCrossover:
                continue
            jlow = np.random.randint( children[0].chromosomeLengths[i] )
            jhigh = np.random.randint( children[0].chromosomeLengths[i] )
            if jlow > jhigh:
                j1low, jhigh = jhigh, jlow
            for j in range( jlow, jhigh + 1 ):
                allele0 = children[0].genome[i][j,0]
                allele1 = children[1].genome[i][j,0]
                j0 = np.where( children[0].genome[i][:,0] == allele1 )[0][0]
                j1 = np.where( children[1].genome[i][:,0] == allele0 )[0][0]
                # first swap genome section of parents
                children[0].genome[i][j,0] = allele1
                children[1].genome[i][j,0] = allele0
                # then swap remaining elements to maintain complete alphabet
                children[0].genome[i][j0,0] = allele0
                children[1].genome[i][j1,0] = allele1
            self.__crossovers += 1

        return children


    def __appendStatistics( self, population ):
        """!
        @brief Append statistics dict of given population.
        """
        fitnessSum = 0.
        fitnessSumSquared = 0.
        fitnessMin = population[0].fitness
        fitnessMax = population[0].fitness
        populationSize = len( population )
        for individual in population:
            fitnessSum += individual.fitness
            fitnessSumSquared += individual.fitness**2
            if individual.fitness > fitnessMax:
                fitnessMax = individual.fitness
            if individual.fitness < fitnessMin:
                fitnessMin = individual.fitness
        statistics = {}
        statistics["mean"] = fitnessSum / populationSize
        statistics["variance"] = fitnessSumSquared / populationSize - \
                                 statistics["mean"]**2
        statistics["min"] = fitnessMin
        statistics["max"] = fitnessMax
        statistics["crossovers"] = self.__crossovers
        statistics["mutations"] = self.__mutations
        statistics["inversions"] = self.__inversions
        if self.__monogamous:
            statistics["divorcerate"] = self.__divorceRate

        self.__statistics.append( statistics )

        return


    def __selection( self, population ):
        """!
        @brief Reduce population size to old populationSize times
               populationGrowth by performing natural selection.
        @param population list to select from as list of genotypes
        @return reduced population
        """
        self.__populationSize = int( round( self.__populationSize *
                                            self.populationGrowth ) )

        if len( population ) == self.__populationSize:
            return population

        retval = []
        indices = sorted( range( len( population ) ),
                          key=lambda k: population[k].fitness,
                          reverse=True )[0:self.__populationSize]

        np.random.shuffle( indices )

        for index in indices:
            retval.append( population[index] )

        return retval


    @property
    def objfunc( self ):
        """!
        @brief Public read-only attribute to obtain the objective function.

        This is NOT a method or member function!
        """
        return self.__objf


    @property
    def decoder( self ):
        """!
        @brief Public read-only attribute to obtain the decoder function.

        This is NOT a method or member function!
        """
        return self.__decf


    @property
    def statistics( self ):
        """!
        @brief Public read-only attribute to obtain list of statistics
               dictionaries.

        The keys into these dictionaries are: "mean", "variance", "min", "max",
        "crossovers", "mutations", and "divorcerate".  Lists of individual
        statistics key can be obtained via [s[key] for s in ga.statistics] where
        ga is a GeneticAlgorithms.Optimizer object.

        It is worth noting that these statistics are computed BEFORE the
        selection of the fittest took place (as they are needed to select them).

        This is NOT a method or member function!
        """
        return self.__statistics


    @property
    def generation( self ):
        """!
        @brief Public read-only attribute to obtain the current generation
               number with 0 being the starting generation.

        This is NOT a method or member function!
        """
        return len( self.__statistics ) - 1


    @property
    def population( self ):
        """!
        @brief Public read-only attribute to obtain the current population of
               Genotype instances.

        This is NOT a method or member function!
        """
        return self.__population


    @property
    def pmx( self ):
        """!
        @brief Public read-only attribute to obtain pmx flag.

        This is NOT a method or member function!
        """
        return self.__pmx


    @property
    def bestFit( self ):
        """!
        @brief Public read-only attribute to obtain decoded parameter and
               fitness of best performing individual in current population as a
               tuple.

        This is NOT a method or member function!
        """
        mf = self.__population[0].fitness
        ind = 0
        for i, individual in enumerate( self.__population ):
            if individual.fitness > mf:
                ind = i
                mf = individual.fitness
        return (self.__population[ind],
                self.__decf( self.__population[ind] ),
                mf)


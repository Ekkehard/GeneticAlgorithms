# Python Implementation: GeneticAlgorithms.
# -*- coding: utf-8 -*-
##
# @file       GeneticAlgorithms.py
#
# @version    1.0.1
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
# @author     Ekkehard Blanz <Ekkehard.Blanz@gmail.com> (C) 2019
#
# @copyright  See COPYING file that comes with this distribution
#
# @mainpage
#
# @b Overview
# @par
#
# The class GeneticAlgorithms from this package implements a set of well-known
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
# The constructor of the class GeneticAlgorithms provides a number of sensible
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
#                   |                |


import multiprocessing
import copy
from threading import Thread
import numpy as np


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


class Genotype( object ):
    """!
    @brief Object representation of a genotype including genome and its
           properties and fitness of the corresponding phenotype in a given
           environment provided by the objective function.  This class is for
           use by user-provided decoder, which receives an instance of this
           class as formal parameter.

    The genome is represented as a list of np-arrays, which can have different
    lengths, each representing one chromosome and its length the number of genes
    in that chromosome.  The chromosomes are single vectors for haploid
    chromosomes and two-column matrices for diploid chromosomes.  All genes are
    encoded as alleles from a given alphabet, which can also be the set of
    floating point numbers.
    """
    def __init__( self, numberChromosomes, chromosomeLengths, chromosomeSets,
                  alleleAlphabet, genome=None, pmx=False ):
        """!
        @brief Constructor sets up a genotype with a random or given genome.
        @param numberChromosomes number of different chromosomes
        @param chromosomeLengths list with lengths of each chromosome
        @param alleleAlphabet alphabet from which to draw alleles
                              (ndarray or type float)
        @param chromosomeSets number of complete chromosome sets (1 or 2)
        @param genome set of chromosomes to use in this class or None
        @param pmx set True to enable partially matched crossover
        """
        self.__genome = []
        self.__chromosomeSets = chromosomeSets
        if chromosomeSets == 2:
            assert( alleleAlphabet is not float )

        if genome is not None:
            self.__genome = copy.deepcopy( genome )
        elif alleleAlphabet is float:
            # the alphabet "float" is obviously a special case
            for i in range( numberChromosomes ):
                self.__genome.append(
                    np.random.random( size=(chromosomeLengths[i],
                                            chromosomeSets) ) )
        else:
            alphabetlen = len( alleleAlphabet )
            chromsize = [(c, chromosomeSets) for c in chromosomeLengths]
            if pmx:
                # start with a random permutation of the alphabet
                for i in range( numberChromosomes ):
                    self.__genome.append(
                        np.random.permutation( alleleAlphabet ).reshape(
                                                                chromsize[i] ) )

            else:
                # start with a set of random elements of the alphabet
                for i in range( numberChromosomes ):
                    self.__genome.append(
                        alleleAlphabet[np.random.randint( alphabetlen,
                                                          size=chromsize[i] )] )

        self.__alphabet = alleleAlphabet
        try:
            self.__characterAlphabet = \
                all( a.isprintable() for a in self.__alphabet )
        except (AttributeError, TypeError):
            self.__characterAlphabet = False
        self.__pmx = pmx

        # no point in providing setters and getters for the following properties
        ## fitness of this genotype in environment provided by objective
        # function
        self.fitness = None
        ## scaled fitness of genotype
        self.scaledFitness = None

        return


    def __str__( self ):
        """!
        @brief Return a string representation of all chromosomes.
        @return string with individual chromosomes separated by commas
        """
        retstr = ""
        if self.alphabet is float or self.__pmx:
            for i in range( self.numberChromosomes ):
                for j in range( self.chromosomeLengths[i] ):
                    retstr += str( self.__genome[i][j,0] )
                    retstr += " "
                retstr = retstr[:-1]
                retstr += ", "
        elif self.haploid:
            for i in range( self.numberChromosomes ):
                for j in range( self.chromosomeLengths[i] ):
                    retstr += str( self.__genome[i][j,0] )
                retstr += ", "
        else:
            for i in range( self.numberChromosomes ):
                retstr += "("
                for j in range( self.chromosomeLengths[i] ):
                    retstr += str( self.__genome[i][j,0] )
                retstr += "),("
                for j in range( self.chromosomeLengths[i] ):
                    retstr += str( self.__genome[i][j,1] )
                retstr += "), "

        return retstr[:-2]


    @property
    def chromosomeLengths( self ):
        """!
        @brief Public read-only attribute to obtain a list with chromosome
               lengths.

        This is NOT a method or member function!
        """
        return [chromosome.shape[0] for chromosome in self.__genome]


    @property
    def haploid( self ):
        """!
        @brief Public read-only attribute to obtain True if chromosomes are
               haploid.

        This is NOT a method or member function!
        """
        return self.__chromosomeSets == 1


    @property
    def diploid( self ):
        """!
        @brief Public read-only attribute to obtain True if chromosomes are
               diploid.

        This is NOT a method or member function!
        """
        return self.__chromosomeSets == 2


    @property
    def numberChromosomes( self ):
        """!
        @brief Public read-only attribute to obtain the number of chromosomes
               in a complete chromosome set.

        This is NOT a method or member function!
        """
        return len( self.__genome )


    @property
    def genome( self ):
        """!
        @brief Public read-only attribute to obtain the full genome as list of
               numpy arrays, each one representing one chromosome.

        This is NOT a method or member function!
        """
        return self.__genome


    @property
    def alphabet( self ):
        """!
        @brief Public read-only attribute to obtain the allele alphabet (could
               be list or type float).

        This is NOT a method or member function!
        """
        return self.__alphabet


    @property
    def alphabetLength( self ):
        """!
        @brief Public read-only attribute to obtain the length of the allele
               alphabet.

        As appropriate, the attribute is np.inf is the allele alphabet is type
        float.

        This is NOT a method or member function!
        """
        if self.alphabet is float:
            return np.inf
        else:
            return len( self.__alphabet )


    @property
    def characterAlphabet( self ):
        """!
        @brief Public read-only attribute to obtain the character alphabet flag.

        This is NOT a method or member function!
        """
        return self.__characterAlphabet


    @property
    def pmx( self ):
        """!
        @brief Public read-only attribute to obtain pmx flag.
        """
        return self.__pmx


class GeneticAlgorithms( object ):
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
        matched crossover (pmx) flag set, they are returned directly as an
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
        @param chromosomeLengths list of lengths of chromosomes
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

        if numberChromosomes == 1 and type( chromosomeLengths ) is not list:
            chromosomeLengths = [chromosomeLengths]
        elif numberChromosomes != len( chromosomeLengths ):
            raise ValueError( "chromosomeLengths must have exactly"
                              "numberChromosomes elements" )

        self.__numberChromosomes = numberChromosomes
        self.__chromosomeLengths = chromosomeLengths

        if pmx:
            if alleleAlphabet is float:
                raise ValueError( "For PMX the alphabet cannot be float" )
            if alleleAlphabet is None:
                alleleAlphabet = np.arange( chromosomeLengths[0] )
            for l in chromosomeLengths:
                if l != len( alleleAlphabet ):
                    raise ValueError( "For PMX the chromosome length must be "
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
                alleleAlphabet = np.array( [0, 1] )
            else:
                alleleAlphabet = np.array( [-1, 0, 1] )
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
            if self.__alleleAlphabet is float or self.__pmx:
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

        for i in range( populationSize ):
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
        for i in range( generations ):
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
            threads = [] * cores
            indices = [] * cores
            threadNo = 0

            for i, individual in enumerate( population ):
                threadNo = i % cores
                if i >= cores:
                    population[indices[threadNo]].fitness = \
                                                        threads[threadNo].join()
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
               rest of  the alleles such that each child is a gain a valid
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
        ga is a GeneticAlgorithms object.

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



if "__main__" == __name__:

    import sys
    import matplotlib.pyplot as plt
    import time


    def printHelp():
        """!
        """
        helpText = \
"""
No help
"""
        print( helpText )
        return 0

    def objfunc0( *args ):
        """!
        @brief Objective function for unit test from Goldberg's book (modified).

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
        @brief Objective function 1 for unit test.

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

        return retval


    def objfunc2( *args ):
        """!
        @brief Objective function 2 for unit test.

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

        return retval


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
        length = len( pwd.pwd )
        correct = 0
        for i in range( length ):
            if pwd.pwd[i] == guess[i]:
                correct += 1
        return correct / length


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
        @brief Interactive demo for GeneticAlgorithms
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
               "objfunc0, objfunc1, objfunc2, and tsp" )
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

        # TODO provide automatic Unit Test
        raise ValueError( "Unit Test not yet implemented" )


    sys.exit( int( main() or 0 ) )

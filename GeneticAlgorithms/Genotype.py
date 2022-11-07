# Python Implementation: Geneotype.py.
# -*- coding: utf-8 -*-
##
# @file       Genotype.py
#
# @version    1.1.1
#
# @par Purpose
# Class to implement the genotype used in Genetic Algorithms including genome 
# and its properties and fitness of the corresponding phenotype in a given
# environment provided by the objective function.  This class is for use by
# user-provided decoder, which receives an instance of this class as formal
# parameter.
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

# File history:
#
#      Date         | Author         | Modification
#  -----------------+----------------+------------------------------------------
#   Tue Aug 20 2019 | Ekkehard Blanz | created
#   Sun Nov 06 2022 | Ekkehard Blanz | extracted fromm GeneticAlgorithms.py
#                   |                |
#

import copy
import numpy as np

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

        # fitness of this genotype in environment provided by objective function
        self.__fitness = None
        # scaled fitness of genotype
        self.__scaledFitness = None

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
    def fitness( self ):
        """!
        @brief Public attribute to obtain fitness of corresponding phenotype.

        This is NOT a method or member function!
        """
        return self.__fitness


    @fitness.setter
    def fitness( self, value ):
        """!
        @brief Setter for fitness property - checks if value is of proper type.

        This is NOT a method or member function!
        """
        try:
            if value < 0:
                raise TypeError
            self.__fitness = value
        except (TypeError, ValueError):
            raise ValueError( "Fitness returned by objective function must be "
                              "a real or integer value\ngreater than or equal "
                              "to 0 - not {0}".format( value ) )
        return


    @property
    def scaledFitness( self ):
        """!
        @brief Public attribute to obtain scaledFitness of corresponding
               phenotype.

        This is NOT a method or member function!
        """
        return self.__scaledFitness


    @scaledFitness.setter
    def scaledFitness( self, value ):
        """!
        @brief Setter for scaledFitness property - checks if value is of proper
               type.

        This is NOT a method or member function!
        """
        try:
            if value < 0:
                raise TypeError
            self.__scaledFitness = value
        except (TypeError, ValueError):
            raise ValueError( "The computed scaled fitness must be "
                              "a real or integer value\ngreater than or equal "
                              "to 0 - not {0}".format( value ) )
        return


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
# Python Implementation: GeneticAlgorithms
# -*- coding: utf-8 -*-
##
# @file       __init__.py
#
# @version    2.0.0
#
"""!
@par Purpose
Genetic Algorithms Toolbox package for Python.

@par Comments
This is Python 3 code!
"""

# Known Bugs: none
#
# @author     Ekkehard Blanz <Ekkehard.Blanz@gmail.com> (C) 2022-2025
#
# @copyright  See COPYING file that comes with this distribution
#
# File history:
#
#      Date         | Author         | Modification
#   ----------------+----------------+------------------------------------------
#   Sun Nov 06 2022 | Ekkehard Blanz | created
#   Wed Nov 30 2022 | Ekkehard Blanz | renamed main class to Optimizer
#   Tue Jan 28 2025 | Ekkehard Blanz | added __version__ and __all__
#                   |                |

from GeneticAlgorithms.Optimizer import Optimizer
from GeneticAlgorithms.Genotype import Genotype

__version__ = '2.0.0'
__all__ = ['Optimizer', 'Genotype']

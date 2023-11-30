import os, sys
location = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(location, 'datasets', 'MdwarfRuns'))
from MdwarfPrediction import Mdwarf_InferPlMass_FromPlRadiusInsolStMass,Mdwarf_InferPlMass_FromPlRadiusStMass,Mdwarf_InferPlMass_FromPlRadius

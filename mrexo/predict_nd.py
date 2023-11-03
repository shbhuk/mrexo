import os, sys
location = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(os.path.dirname(location), 'sample_scripts', 'MdwarfRuns'))
from MdwarfPrediction import Mdwarf_InferPlMass_FromPlRadiusInsolStMass,Mdwarf_InferPlMass_FromPlRadiusStMass,Mdwarf_InferPlMass_FromPlRadius

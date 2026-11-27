'''
    Author:     Matthew Ogden and Graham West
    Created:    28 Mar 2022
Description:    This code facilitates the Genetic Algorithm for evolving Galactic Models.
                Large portions of this code are copied or derived from Graham West's code found at the following link.
                https://github.com/gtw2i/GA-Galaxy
'''

# For loading in Matt's general purpose python libraries
from os import path
from sys import path as sysPath
supportPath = path.abspath( path.join( __file__ , "../../Support_Code/" ) )
sysPath.append( supportPath )
import general_module as gm
import info_module as im

# For loading other dependencies
from copy import deepcopy
import numpy as np
import numpy.linalg as LA
import math
import random
import pickle

# Generate random instance
rng = np.random.default_rng()


def test():
    print("GA: Hi!  You're in Matthew's main code for all things genetric algorithm.")


def Genetic_Algorithm_Experiment( ga_param, scorerFunc, \
                                 writeLocBase, printProg = False ):

    print("GA.Genetic_Algorithm_Experiment: Beginning Experiment")
    
    # RUN GA
    pFit = ga_param['parameter_to_fit'] 
    pReal = ga_param['parameter_fixed_values'] 
    nPhase = ga_param['phase_number'] 
    tGen   = ga_param['generation_total_number'] 
    phaseParams = ga_param['phase_parameter_to_fit'] 
    nFits = len( phaseParams )
    
    # Variable for printing progress
    pVal = { 'phase':0, 'gen':0, 'tGen':tGen }

    # Fit all parameters once before entering phases.
    print('Phase: %d / %d: Gen: %d\n' % ( pVal['phase'], nPhase, pVal['gen'] ) )
    chain, scores, pVal = Genetic_Algorithm_Phase( pFit, pReal, scorerFunc, ga_param, printProg = printProg, pVal=pVal )
    
    print("chain: ", chain.shape)
    print("nPhase", nPhase)
    print("genNumber", ga_param['generation_number'])

    # Save initial progress
    pickle.dump( chain,   open( writeLocBase + "models.pkl", "wb" ) )
    pickle.dump( scores,  open( writeLocBase + "scores.pkl", "wb" ) )

    # Loop through desired phases. 
    for phase in range( 1, nPhase ): 

        pVal['phase'] += 1
        print('Phase: %d / %d: Gen: %d\n' % ( pVal['phase'], nPhase, pVal['gen'] ) )

        # Get best solution from previous phase
        maxScoInd = np.unravel_index( np.argmax(scores, axis=None), scores.shape )
        pBest = chain[maxScoInd[0],maxScoInd[1],:]

        # Get list of parameters to fit. 
        pFit = phaseParams[ (phase-1) % nFits ]

        # RUN GA, previous best model as target
        chain1, scores1, pVal = Genetic_Algorithm_Phase( pFit, pBest, scorerFunc, ga_param, chain, scores, printProg = printProg, pVal=pVal )

        # Add new data to previous data
        chain  = np.concatenate( ( chain,  chain1  ) )
        scores = np.concatenate( ( scores, scores1 ) )
        
        print(chain.shape)

        # Save models and scores obtained until now
        pickle.dump( chain,   open( writeLocBase + "models.pkl", "wb" ) )
        pickle.dump( scores,  open( writeLocBase + "scores.pkl", "wb" ) )

    print("********************************************")
    print("*  GA.Genetic_Algorithm_Experiment: DONE!  *")
    print("********************************************")

def Genetic_Algorithm_Phase( pFit, start, scoreModels, ga_param, \
                            old_chain = None, old_scores = None, \
                            printProg = True, printAll = False, \
                            pVal = { 'phase':0, 'gen':0 } ) :
    
    # Grab needed values from parameter file 
    nGen   = ga_param['generation_number'] 
    nParam = ga_param['parameter_number']
    xLim   = ga_param['parameter_limits']
    psi    = ga_param['parameter_psi']
    nPop   = ga_param['population_size']
    nKeep  = ga_param['population_keep']
    reseed = ga_param['population_reseed_ratio']
    toMix  = ga_param['covariance_mix_matrix']
    burn   = ga_param['covariance_burn']
    mixAmp = ga_param['covariance_mix_amplitude']
    mixProb = ga_param['covariance_mix_probability']
    sigScale = ga_param['covariance_scale']

    # Initialize variables for convariance matrix
    stds = np.zeros(nParam)
    for i in range(nParam):
        if i in pFit:
            stds[i] = xLim[i,1]-xLim[i,0]
        else:
            stds[i] = 0.00001
        # end
    # end
    
    pWidth  = stds*sigScale
    covInit = np.diag(pWidth**2)
    cov     = np.diag(pWidth**2)
    cov2    = np.diag(pWidth**2)
    C       = np.diag(pWidth**2)
    mean    = deepcopy(start)[0:14]

    # get initial population
    
    if printProg: 
        print("GA: Gen: %d Phase %d: Initial" % \
              (pVal['gen'], pVal['phase'], ) )
    pVal['gen'] += 1
        
    popSol = getInitPop( nPop, start, pFit, ga_param )
    
    # Preserve top models from last phase.    
    if type( old_chain ) != type( None ) and nKeep > 0:
        
        # Grab indices of top models
        top_ind = np.argpartition( old_scores[-1,:], -nKeep)[-nKeep:]   
        
        # Copy top models
        popSol[0:nKeep,:] = old_chain[-1,top_ind,:]
        
        # Copy scores
        popFit = np.zeros(old_scores[-1,:].shape)
        popFit[0:nKeep] = old_scores[-1,top_ind]
        
        # Score rest of models
        popFit[nKeep + 1:] = scoreModels( popSol[nKeep+1:,:] )
        
        
        # Print results
        print('old scores: ', old_scores[-1,:])
        print('new scores: ', popFit)
              
    else:
        popFit = scoreModels( popSol ) 
    # end
    
    # get best solution from initalial population
    fBest   = np.max(popFit)
    bestInd = np.argmax(popFit)
    pBest   = popSol[bestInd,:]
    
    chain = [ popSol ]
    fit   = [ popFit ]

    adaCount = 0

    chainF = []
    for i in range(nPop):
        chainF.append( popSol[i] )
    # end

    # Random walk
    for step in range(1, nGen):
        
        if printProg: 
            print("GA: Gen: %d Phase %d: Step: %d/ %d" % \
                  (pVal['gen'], pVal['phase'], step, nGen) )
            pVal['gen'] += 1

        if( step > 0 and reseed > 0):
            ind = max( 1, int(reseed*nPop) )
            popSol_re = getInitPop( ind, start, pFit, ga_param)
            order = np.argsort(popFit)
            popSol[order[:ind],:] = popSol_re
        # end

        # perform selection
        parSol, parFit = Selection( nPop, popSol, popFit, nKeep )

        # perform crossover
        popSol = Crossover( nPop, nParam, parSol, parFit, cov, step, nGen )

        # get covariance matrix
        #cov, mean, C = getHaarioCov( covInit, C, mean, step, burn, nPop, nParam, chainF, popSol, pFit )
        
        # If mix matrix given
        if toMix:
            cov2 = mixCov( cov, covInit, step, burn, nPop, nParam, mixProb, mixAmp )
        else:
            cov2 = deepcopy( cov )

        # perform mutation
        popSol = Mutate( step, nGen, nPop, nParam, popSol, cov2, xLim, pFit, nKeep )

        # calculate fits
        if printAll: print("GA: Scoring Models\n")
        
        # Last "nKeep" values are best models from prev gen.
        popFit = np.zeros(popFit.shape)
        popFit[-nKeep:] = parFit[-nKeep:,0]
    
        # Score new models
        popFit[:nPop-nKeep] = scoreModels( popSol[:nPop-nKeep] )
        
        # get best solution
        fTest = np.max(popFit)
        if( fBest < fTest ):
            fBest   = fTest
            bestInd = np.argmax(popFit)
            pBest   = popSol[bestInd,:]
        # end

        chain.append( popSol )
        fit.append(   popFit )
        for i in range(nPop):
            chainF.append( popSol[i] )
        # end

        if printAll:
            print(np.amax(popFit))
            print(popSol[np.argmax(popFit),:])
    # end random walk
    
    chain = np.array(chain)
    fit   = np.array(fit)

    if printAll: print(" ")

    return chain, fit, pVal
# End Generation_Phase


def getInitPop( nPop, pReal, pFit, ga_param ):
    
    # Grab needed variables from input.
    nParam = ga_param['parameter_number']
    xLim   = ga_param['parameter_limits']
    initType = ga_param['population_initialization_method']
    nSample  = ga_param['population_sample_size'] 

    # Create blank array for model population to fill
    popSol = np.zeros((nPop,nParam))

    if initType == 0:        
        # Loop through each new model
        for j in range(nParam):
            if j in pFit:
                popSol[:,j] = np.random.uniform( xLim[j,0], xLim[j,1], nPop )
            else:
                popSol[:,j] = pReal[j] * np.ones( nPop )
    # end initType 0
        
    elif initType == 1:
        
        maxCorr = 0.0
        
        # Loop through N sample populations.
        for i in range( nSample ):
            
            # Generate a model population with shuffled evenly spaced variable values.
            R = np.zeros((nParam, nPop))
            for j in range(nParam):
                
                tmp = np.linspace( xLim[j,0], xLim[j,1], nPop )
                rng.shuffle( tmp )
                R[j,:] = tmp

            # Find model population with greatest "model space"
            corr = LA.det(np.corrcoef(R[pFit,:]))
            if( corr > maxCorr or i == 0 ):
                maxCorr = corr
                maxR = deepcopy(R)
        
        # end N samples. 

        # Save model population with the greatest "model space"
        for i in range(nPop):
            for j in range(nParam):
                if j in pFit:
                    popSol[i,j] = maxR[j,i]
                else:
                    popSol[i,j] = pReal[j]
        
    # end initType 1
    
    else:
        print("ERROR: GA.getInitPop:")
        gm.tabprint("Population Init Type not available: %s" % str(initType))
        popSol = None

    return popSol

# end getInitPop


# NOTE: The following converts the SPAM parameters into one of 8 true symmetries Graham identies. 
def convert_spam_to_ga( in_param ):
    
    if len( in_param.shape ) == 1:
        in_param = np.expand_dims(in_param, axis=0)
    
    out_param = deepcopy(in_param[:,0:14])
    psi = []
    
    for i in range( out_param.shape[0]):

        psi_p = 1
        psi_s = 1

        if( out_param[i,2] < 0 ):
            out_param[i,2]  =  -1 * out_param[i,2]
            out_param[i,5]  =  -1 * out_param[i,5]
            out_param[i,10] = 180 + out_param[i,10]
            out_param[i,11] = 180 + out_param[i,11]
        # end
        out_param[i,10] %= 360
        out_param[i,11] %= 360
        if( out_param[i,10] > 180 ):
            out_param[i,10] = out_param[i,10] - 180
            out_param[i,12] = -1 * out_param[i,12]
        # end
        if( out_param[i,11] > 180 ):
            out_param[i,11] = out_param[i,11] - 180
            out_param[i,13] = -1 * out_param[i,13]
        # end
        out_param[i,12] %= 360
        out_param[i,13] %= 360
        if( out_param[i,12] > 180 ):
            out_param[i,12] = out_param[i,12] - 360
        # end
        if( out_param[i,13] > 180 ):
            out_param[i,13] = out_param[i,13] - 360
        # end

        if( out_param[i,12] > 90 ):
            out_param[i,12] = out_param[i,12] - 180
            psi_p = -1
        elif( out_param[i,12] < -90 ):
            out_param[i,12] = out_param[i,12] + 180
            psi_p = -1
        # end
        if( out_param[i,13] > 90 ):
            out_param[i,13] = out_param[i,13] - 180
            psi_s = -1
        elif( out_param[i,13] < -90 ):
            out_param[i,13] = out_param[i,13] + 180
            psi_s = -1
            
        # end
        psi.append( [psi_p,psi_s] )
    # end for loop
    
    psi = np.array(psi)

    # energy
    G = 1
    r = ( out_param[:,0]**2 + out_param[:,1]**2 + out_param[:,2]**2 )**0.5
    U = -G*out_param[:,6]*out_param[:,7]/r
    v = ( out_param[:,3]**2 + out_param[:,4]**2 + out_param[:,5]**2 )**0.5
    K = 0.5*out_param[:,7]*v**2
    c = (K+U)/(K-U)

    # convert p,s mass to ratio,total mass
    t = out_param[:,6] + out_param[:,7]
    f = out_param[:,6] / t
    out_param[:,6] = f
    out_param[:,7] = t

    # spherical velocity
    phi   = ( np.arctan2( out_param[:,4], out_param[:,3] ) * 180.0 / np.pi ) % 360
    theta = ( np.arcsin( out_param[:,5] / v ) * 180.0 / np.pi )

    out_param[:,3] = c

    out_param[:,4] = phi
    out_param[:,5] = theta

    Ap = np.abs(out_param[:,8]**2*np.cos(out_param[:,12]*np.pi/180.0))
    As = np.abs(out_param[:,9]**2*np.cos(out_param[:,13]*np.pi/180.0))

    out_param[:,8] = Ap
    out_param[:,9] = As

    return out_param, psi

# End convert_spam_to_ga


def convert_ga_to_spam( in_param, psi ):
    
    # Expand in parameters if a single vector
    if len( in_param.shape ) == 1:
        in_param = np.expand_dims(in_param, axis=0)
    
    # Create seperate matrix for output parameters
    out_param = deepcopy(in_param)
    
    for i in range( out_param.shape[0]):
        
        # add for psi
        if( psi[0] == -1 ):
            out_param[i,12] += 180.0
        # end
        if( psi[1] == -1 ):
            out_param[i,13] += 180.0
        # end

        # convert mass units
        f    = out_param[i,6]
        t    = out_param[i,7]
        out_param[i,6] = f*out_param[i,7]
        out_param[i,7] = (1.0-f)*out_param[i,7]

        G = 1
        c = out_param[i,3]
        v = ( (1+c)/(1-c)*2*G*out_param[i,6]/(out_param[i,0]**2+out_param[i,1]**2+out_param[i,2]**2)**0.5 )**0.5    

        vx = v*math.cos(out_param[i,4]*np.pi/180.0)*math.cos(out_param[i,5]*np.pi/180.0)
        vy = v*math.sin(out_param[i,4]*np.pi/180.0)*math.cos(out_param[i,5]*np.pi/180.0)
        vz = v*math.sin(out_param[i,5]*np.pi/180.0)
        out_param[i,3] = vx
        out_param[i,4] = vy
        out_param[i,5] = vz
        
        Ap = out_param[i,8]
        As = out_param[i,9]
        out_param[i,8] = np.abs(Ap/np.cos(out_param[i,12]*np.pi/180.0))**0.5
        out_param[i,9] = np.abs(As/np.cos(out_param[i,13]*np.pi/180.0))**0.5
    
    
    if out_param.shape[1] < 15:
        diff = 15 - out_param.shape[1]
        out_param = np.pad( out_param, [(0, 0), (0, diff)], mode='constant')
        
    return out_param

# End convert_ga_to_spam


def rotMat( axis, theta ):
    theta = theta * np.pi / 180.0
    axis = axis/LA.norm(axis)
    a = math.cos( theta / 2.0 )
    b, c, d = -axis * math.sin( theta / 2.0 )
    aa, bb, cc, dd = a**2, b**2, c**2, d**2
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    M = [	[ aa+bb-cc-dd, 2*(bc+ad), 2*(bc-ad) ],
        [ 2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab) ],
        [ 2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc ] ]

    return np.array(M)
# end

def Selection( nPop, popSol, popFit, nKeep ):
    
    '''
    print("Selection")
    print("NPop", nPop)
    print("popSol", popSol.shape, popSol)
    print("popFit", popFit.shape, popFit)
    print("nKeep", nKeep)
    '''

    # 0: fitness-squared proportional selection
    # 1: rank proportional selection
    selectType = 0

    parSol = []
    parFit = []

    if( selectType == 0 ):
        xxx = popFit**1
    #		xxx = np.ones(nPop)
        xxx = xxx/np.sum(xxx)
        popProb = np.cumsum( xxx )

        for i in range(nPop-nKeep):
            r1 = np.random.uniform(0,1)
            r2 = np.random.uniform(0,1)

            ind1 = np.argmax( r1 <= popProb )
            ind2 = np.argmax( r2 <= popProb )
            
            parSol.append( [ popSol[ind1], popSol[ind2] ] )
            parFit.append( [ popFit[ind1], popFit[ind2] ] )
        # end

        srt = np.argsort( popFit )
        for i in range(1,nKeep+1):
            parSol.append( [ popSol[srt[-i]], popSol[srt[-i]] ] )
            parFit.append( [ popFit[srt[-i]], popFit[srt[-i]] ] )
        # end
    else:
        # get selection probabilities
        inds = popFit.argsort() + np.ones(nPop)
        popProb = np.cumsum( inds/np.sum(inds) )

        for i in range(nPop):
            r1 = np.random.uniform(0,1)
            r2 = np.random.uniform(0,1)

            ind1 = np.argmax( r1 <= popProb )
            ind2 = np.argmax( r2 <= popProb )
            
            parents.append( [ popSol[ind1], popSol[ind2] ] )
        # end

    # end
    parSol = np.array(parSol)
    parFit = np.array(parFit)

    return parSol, parFit
    # end

def Crossover( nPop, nParam, parSol, parFit, cov, step, burn ):

    # 0: parameter swapping w/ PCA
    # 1: mean crossover
    """
    if( step < burn ):
        crossType = 1
    else:
        crossType = 0
    # end
    """
    crossType = 0

    popSol = np.zeros((nPop,nParam))

    if(   crossType == 0 ):
        if( step < burn ):
            for i in range(nPop):
                for j in range(nParam):
                    r0 = parFit[i,0]/(parFit[i,0]+parFit[i,1])
                    r  = np.random.uniform(0,1)

                    if( r < r0 ):
                        popSol[i,j] = parSol[i,0,j]
                    else:
                        popSol[i,j] = parSol[i,1,j]
                    # end
                # end
            # end
        else:
            # get PCA
            w, v = LA.eig(cov)

            # convert to PCA basis
            pcaPar = []
            for i in range(nPop):
                p1 = np.dot( v, parSol[i,0,:] )
                p2 = np.dot( v, parSol[i,0,:] )
                pcaPar.append( [p1, p2] )
            # end
            pcaPar = np.array(pcaPar)

            # mix in PCA basis
            for i in range(nPop):
                for j in range(nParam):
                    r0 = parFit[i,0]/(parFit[i,0]+parFit[i,1])
                    r  = np.random.uniform(0,1)

                    if( r < r0 ):
                        popSol[i,j] = pcaPar[i,0,j]
                    else:
                        popSol[i,j] = pcaPar[i,1,j]
                    # end
                # end
            # end

            # convert back to parameter basis
            vinv = LA.inv(v)
            for i in range(nPop):
                popSol[i,:] = np.dot( vinv, popSol[i,:] )
            # end
        # end
        popSol = np.array(popSol)
    elif( crossType == 1 ):
        for i in range(nPop):
            r0 = parFit[i,0]/(parFit[i,0]+parFit[i,1])
            c1 = parSol[i,0,:]*r0 + parSol[i,1,:]*(1-r0)
            popSol[i,:] = c1
        # end
    # end

    return popSol
# end

def Mutate( step, nGen, nPop, nParam, popSol, cov, xLim, pFit, nKeep ):

    popSol2 = np.zeros((nPop,nParam))

    #	abc = np.linspace( xLim[2,0], xLim[2,1], nPop )
    #	random.shuffle(abc)

    for i in range(nPop):

        # perturb
        x = np.random.multivariate_normal(mean=popSol[i,:],cov=cov,size=1)[0]

        # sphere walk, prim
        phi_v1   = popSol[i,4]/180.0*np.pi
        theta_v1 = popSol[i,5]/180.0*np.pi

        e_r_v   = np.array( [  math.cos(phi_v1)*math.cos(theta_v1), math.sin(phi_v1)*math.cos(theta_v1), math.sin(theta_v1) ] )
        e_phi_v = np.array( [                  -math.sin(phi_v1),                  math.cos(phi_v1),              0.0 ] )

        a_v = np.abs( np.random.normal( 0.0, cov[4,4]**0.5 ) )
        b_v = np.random.uniform(0.0,360.0)

        rot_e_r_v = rotMat( e_r_v, b_v )
        e_phi_rot_v = np.dot( rot_e_r_v, e_phi_v )

        cross_v = np.cross( e_r_v, e_phi_rot_v )

        rot_cross_v = rotMat( cross_v, a_v )
        e_r_rot_v = np.dot( rot_cross_v, e_r_v )

        phi_v2   = np.arctan2( e_r_rot_v[1], e_r_rot_v[0] )*180.0/np.pi
        theta_v2 = np.arcsin(  e_r_rot_v[2] )*180.0/np.pi

        x[4] = phi_v2 % 360.0
        x[5] = theta_v2

        # sphere walk, prim
        phi_p1   = popSol[i,10]/180.0*np.pi
        theta_p1 = popSol[i,12]/180.0*np.pi

        e_r_p   = np.array( [  math.cos(phi_p1)*math.cos(theta_p1), math.sin(phi_p1)*math.cos(theta_p1), math.sin(theta_p1) ] )
        e_phi_p = np.array( [                  -math.sin(phi_p1),                  math.cos(phi_p1),              0.0 ] )

        a_p = np.abs( np.random.normal( 0.0, cov[10,10]**0.5 ) )
        b_p = np.random.uniform(0.0,360.0)

        rot_e_r_p = rotMat( e_r_p, b_p )
        e_phi_rot_p = np.dot( rot_e_r_p, e_phi_p )

        cross_p = np.cross( e_r_p, e_phi_rot_p )

        rot_cross_p = rotMat( cross_p, a_p )
        e_r_rot_p = np.dot( rot_cross_p, e_r_p )

        phi_p2   = np.arctan2( e_r_rot_p[1], e_r_rot_p[0] )*180.0/np.pi
        theta_p2 = np.arcsin(  e_r_rot_p[2] )*180.0/np.pi

        x[10] = phi_p2 % 360.0
        x[12] = theta_p2

        # sphere walk, sec
        phi_s1   = popSol[i,11]/180.0*np.pi
        theta_s1 = popSol[i,13]/180.0*np.pi

        e_r_s   = np.array( [  math.cos(phi_s1)*math.cos(theta_s1), math.sin(phi_s1)*math.cos(theta_s1), math.sin(theta_s1) ] )
        e_phi_s = np.array( [                  -math.sin(phi_s1),                  math.cos(phi_s1),              0.0 ] )

        a_s = np.abs( np.random.normal( 0.0, cov[11,11]**0.5 ) )
        b_s = np.random.uniform(0.0,360.0)

        rot_e_r_s = rotMat( e_r_s, b_s )
        e_phi_rot_s = np.dot( rot_e_r_s, e_phi_s )

        cross_s = np.cross( e_r_s, e_phi_rot_s )

        rot_cross_s = rotMat( cross_s, a_s )
        e_r_rot_s = np.dot( rot_cross_s, e_r_s )

        phi_s2   = np.arctan2( e_r_rot_s[1], e_r_rot_s[0] )*180.0/np.pi
        theta_s2 = np.arcsin(  e_r_rot_s[2] )*180.0/np.pi

        x[11] = phi_s2 % 360.0
        x[13] = theta_s2

    #		print xLim[4,0], x[4], xLim[4,1]

        # bounds check
        eps = 0.00001
        for j in range(nParam):
            if( i >= nPop-nKeep ):
                popSol2[i,j] = popSol[i,j]
            else:
                if(   xLim[j,0] > x[j] and j in pFit ):
                    popSol2[i,j] = 0.5*( xLim[j,0] + popSol[i,j] )
    #					popSol2[i,j] = xLim[j,0] + eps*(xLim[j,1]-xLim[j,0])
                elif( xLim[j,1] < x[j] and j in pFit ):
                    popSol2[i,j] = 0.5*( xLim[j,1] + popSol[i,j] )
    #					popSol2[i,j] = xLim[j,1] - eps*(xLim[j,1]-xLim[j,0])
                else:
                    popSol2[i,j] = x[j]
                # end
            # end
        # end

    # end

    return popSol2
# end

def getHaarioCov( covInit, C, mean, step, burn, nPop, nParam, chainF2, popSol, pFit ):

    chainF = np.array(chainF2)

    # get AM cov matrix
    if( step < burn ):
        C    = deepcopy(covInit)
        cov2 = deepcopy(covInit)
    elif( step == burn ):
        mean = np.mean( np.array(chainF)[int(nPop*burn/2):,:], axis=0 )
        C    = np.cov( np.transpose( np.array(chainF)[int(nPop*burn/2):,:] ) )

        C2   = C[np.ix_(pFit,pFit)]
        w, v = LA.eig(C2)
    #		cov2  = C*( np.abs( np.prod(np.diag(covInit)[pFit]) / np.prod(w[pFit]) ) )**(1.0/len(pFit))
        cov2  = C*( np.abs( np.prod(np.diag(covInit)[pFit]) / np.prod(w) ) )**(1.0/len(pFit))
        cov2[0,:] = covInit[0,:]
        cov2[1,:] = covInit[1,:]
        cov2[7,:] = covInit[7,:]
        cov2[:,0] = covInit[:,0]
        cov2[:,1] = covInit[:,1]
        cov2[:,7] = covInit[:,7]
    elif( step > burn ):
        for i in range(nPop):
            gamma = 1.0/(nPop*(step+1)+i)
            dx   = popSol - mean
            mean = mean + gamma*dx
            C    = C    + gamma*(np.outer(dx,dx) - C)
        # end

        C2   = C[np.ix_(pFit,pFit)]
        w, v = LA.eig(C2)
    #		cov2  = C*( np.abs( np.prod(np.diag(covInit)[pFit]) / np.prod(w[pFit]) ) )**(1.0/len(pFit))
        cov2  = C*( np.abs( np.prod(np.diag(covInit)[pFit]) / np.prod(w) ) )**(1.0/len(pFit))
        cov2[0,:] = covInit[0,:]
        cov2[1,:] = covInit[1,:]
        cov2[7,:] = covInit[7,:]
        cov2[:,0] = covInit[:,0]
        cov2[:,1] = covInit[:,1]
        cov2[:,7] = covInit[:,7]
    # end

    return cov2, mean, C
# end

def mixCov( cov, covInit, step, burn, nPop, nParam, mixProb, mixAmp ):

    cov2 = deepcopy(cov)
    
    # apply mixing
    if(   step < burn ):
        # scale
        for i in range(nParam):
            s = np.random.uniform(0,1)

            # thinning
            if( s <= mixProb[0] ):
                cov2[i,i] = covInit[i,i]*mixAmp[0]**2
            # widening
            elif( s <= mixProb[0] + mixProb[1] ):
                cov2[i,i] = covInit[i,i]*mixAmp[1]**2
            # fixing
            else:
                cov2[i,i] = covInit[i,i]
            # end
        # end
    elif( step >= burn ):
        # decompose, normalize
        w, v = LA.eig(cov)

        # mix
        for i in range(nParam):
            s = np.random.uniform(0,1)

            # thinning
            if( s <= mixProb[0] ):
                w[i] = w[i]*mixAmp[0]**2
            # widening
            elif( s <= mixProb[0] + mixProb[1] ):
                w[i] = w[i]*mixAmp[1]**2
            # fixing
            else:
                w[i] = w[i]
            # end
        # end

        # recompose matrix
        W    = np.diag(w)
#			cov2 = np.dot( np.dot(v,W), LA.inv(v) )
        cov2 = np.dot( np.dot(v,W), np.transpose(v) )
        cov2 = cov2.real
        cov2 = 0.5*( cov2 + np.transpose(cov2) )
    # end

    return cov2
# end



def ReadAndCleanupData( filePath, thresh ):
    
    import pandas as pd

    print("Cleaning target file...")

    # read data into np array
    df = pd.read_csv( filePath, sep=',|\t', engine='python', header=None )
    data1 = df.values

    # remove unranked models
    ind = 0
    while( not math.isnan(data1[ind,-1]) ):
        ind += 1
    # end
    data2 = data1[0:ind,:]
    nModel = ind + 1

    # include human score and SPAM params
    cols = list(range(4,18)) + [ 1 ]
    data2 = data2[:,cols]

    # ignore bad zoo models
    data2 = data2[data2[:,-1]>=thresh,:]
    nModel = data2.shape[0]

    data2 = np.array( data2, dtype=np.float32 )

#	data2[0,2]  = 10
#	data2[0,5]  = -10
#	data2[0,10] = 20
#	data2[0,11] = 20
#	data2[0,12] = 100
#	data2[0,13] = 100

    data3 = deepcopy(data2)

    psi = []
    for i in range(nModel):
        psi_p = 1
        psi_s = 1

        if( data2[i,2] < 0 ):
            data2[i,2]  =  -1 * data2[i,2]
            data2[i,5]  =  -1 * data2[i,5]
            data2[i,10] = 180 + data2[i,10]
            data2[i,11] = 180 + data2[i,11]
        # end
        data2[i,10] %= 360
        data2[i,11] %= 360
        if( data2[i,10] > 180 ):
            data2[i,10] = data2[i,10] - 180
            data2[i,12] = -1 * data2[i,12]
        # end
        if( data2[i,11] > 180 ):
            data2[i,11] = data2[i,11] - 180
            data2[i,13] = -1 * data2[i,13]
        # end
        data2[i,12] %= 360
        data2[i,13] %= 360
        if( data2[i,12] > 180 ):
            data2[i,12] = data2[i,12] - 360
        # end
        if( data2[i,13] > 180 ):
            data2[i,13] = data2[i,13] - 360
        # end

        if( data2[i,12] > 90 ):
            data2[i,12] = data2[i,12] - 180
            psi_p = -1
        elif( data2[i,12] < -90 ):
            data2[i,12] = data2[i,12] + 180
            psi_p = -1
        # end
        if( data2[i,13] > 90 ):
            data2[i,13] = data2[i,13] - 180
            psi_s = -1
        elif( data2[i,13] < -90 ):
            data2[i,13] = data2[i,13] + 180
            psi_s = -1
        # end
        psi.append( [psi_p,psi_s] )
    # end
    psi = np.array(psi)
    

    # energy
    G = 1
    r = ( data2[:,0]**2 + data2[:,1]**2 + data2[:,2]**2 )**0.5
    U = -G*data2[:,6]*data2[:,7]/r
    v = ( data2[:,3]**2 + data2[:,4]**2 + data2[:,5]**2 )**0.5
    K = 0.5*data2[:,7]*v**2
#	c = np.log(1-K/U)
    c = (K+U)/(K-U)

    # convert p,s mass to ratio,total mass
    t = data2[:,6] + data2[:,7]
    f = data2[:,6] / t
    data2[:,6] = f
    data2[:,7] = t

    # spherical velocity
    phi   = ( np.arctan2( data2[:,4], data2[:,3] ) * 180.0 / np.pi ) % 360
    theta = ( np.arcsin( data2[:,5] / v ) * 180.0 / np.pi )

#	data2[:,2] = c
#	data2[:,3] = v
    data2[:,3] = c
#	data2[:,2] = K
#	data2[:,3] = U

    data2[:,4] = phi
    data2[:,5] = theta

#	"""
    Ap = np.abs(data2[:,8]**2*np.cos(data2[:,12]*np.pi/180.0))
    As = np.abs(data2[:,9]**2*np.cos(data2[:,13]*np.pi/180.0))

    data2[:,8] = Ap
    data2[:,9] = As
#	"""

    return data2, psi, nModel, len(cols)

# end ReadandCleanUpData


def Prep_GA_Input_Parameters( tInfo, pLoc ):

    # Check if file exists
    pLoc = gm.validPath( pLoc )
    if pLoc == None:
        return None
    
    # Read file
    ga_param = gm.readJson( pLoc )
    if type(ga_param) == type(None):
        return None

    # Do some preprocessing of variables.
    ga_param['generation_number'] = int( ga_param['generation_total_number'] / ga_param['phase_number'] )
    
    ga_param['covariance_burn'] = np.power( ga_param['generation_number']/2, -1 )
    ga_param['covariance_mix_probability'] = ga_param['covariance_mix_probability']/np.sum(ga_param['covariance_mix_probability'])
    
    # Get values based on models from Galaxy Zoo: Mergers
    if ga_param.get('parameter_limits',None) == None:
        
        zoo_params = initialize_ga_parameters(tInfo)
        
        ga_param['parameter_fixed_values'] = zoo_params['parameter_fixed_values']
        ga_param['parameter_psi']          = zoo_params['parameter_psi']
        ga_param['parameter_limits']       = zoo_params['parameter_limits']
    
    else:
        ga_param['parameter_limits'] = np.array( ga_param['parameter_limits'] )

    return ga_param


def initialize_ga_parameters( tInfo, ):
    
    print('GA.initialize_ga_parameters')
    
    # Intialize parameter variable
    ga_param = {}
    ga_param['name'] = 'init'
    ga_param['target_id'] = tInfo.get('target_id')
    
    # Read Orbital Paramters of Galaxy Zoo Merger Models
    #print( )
    n = len( tInfo.tDict['zoo_merger_models'].keys() )
    spam_parameters = np.zeros((n,14))
    for i, rId in enumerate(tInfo.tDict['zoo_merger_models']):
        
        modelStr = tInfo.tDict['zoo_merger_models'][rId]['model_data']
        for j,val in enumerate( modelStr.split(',') ):
            spam_parameters[i,j] = float(val)
            if j == 13: break
    
    # Convert parameters to GA coordinate system
    ga_parameters, ga_psi = convert_spam_to_ga( spam_parameters )
    print(ga_parameters.shape)
    
    # Use top model for psi and fixed values
    ga_param['parameter_psi'] = ga_psi[0]
    ga_param['parameter_fixed_values'] = ga_parameters[0,:]
    
    # Calculate parameter limits
    ga_param['parameter_limits'] = np.zeros((14,2))
    mins = np.min( ga_parameters, axis=0 )
    maxs = np.max( ga_parameters, axis=0 )
    
    # Deafult limits are the min/max found in galaxy zoo Mergers
    for i in range(14):
        ga_param['parameter_limits'][i,:] = np.array( [ mins[i], maxs[i] ] )
    
    # Hardcoded limits
    ga_param['parameter_limits'][2,0] = 0  # sym
    ga_param['parameter_limits'][10,:] = np.array([   0.0, 180.0 ])  # sym
    ga_param['parameter_limits'][11,:] = np.array([   0.0, 180.0 ])  # sym
    ga_param['parameter_limits'][12,:] = np.array([ -89.0,  89.0 ])
    ga_param['parameter_limits'][13,:] = np.array([ -89.0,  89.0 ])
    
    return ga_param

# end Create_Parameter_Limits


# Run main after declaring functions
if __name__ == '__main__':
    from sys import argv
    arg = gm.inArgClass( argv )
    main( arg )

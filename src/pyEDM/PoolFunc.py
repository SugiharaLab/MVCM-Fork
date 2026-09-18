
# python modules

# package modules

# local modules
import pyEDM.API as API
from .AuxFunc import ComputeError
from scipy.stats import spearmanr, pearsonr
import numpy as np
import pandas as pd

#------------------------------------------------------
# Function to evaluate multiview predictions top combos
#------------------------------------------------------
def MultiviewSimplexPred( combo, data, args ) :

    df = API.Simplex( dataFrame       = data,
                      columns         = list( combo ),
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = args['E'], 
                      Tp              = args['Tp'],
                      tau             = args['tau'],
                      exclusionRadius = args['exclusionRadius'],
                      embedded        = args['embedded'],
                      noTime          = args['noTime'],
                      kdWorkers       = args['kdWorkers'],
                      ignoreNan       = args['ignoreNan'] )
    return df

#----------------------------------------------------
# Function to evaluate combo rank (rho)
#----------------------------------------------------
def MultiviewSimplexRho( combo, data, args ) :

    df = API.Simplex( dataFrame       = data,
                      columns         = list( combo ),
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = args['E'], 
                      Tp              = args['Tp'],
                      tau             = args['tau'],
                      exclusionRadius = args['exclusionRadius'],
                      embedded        = args['embedded'],
                      noTime          = args['noTime'],
                      ignoreNan       = args['ignoreNan'] )

    err = ComputeError( df['Observations'], df['Predictions'] )
    return err['rho']

#----------------------------------------------------
# Function to evaluate Simplex in EmbedDimension Pool
#----------------------------------------------------
def EmbedDimSimplexFunc( E, data, args ) :

    df = API.Simplex( dataFrame       = data,
                      columns         = args['columns'],
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = E, 
                      Tp              = args['Tp'],
                      tau             = args['tau'],
                      exclusionRadius = args['exclusionRadius'],
                      embedded        = args['embedded'],
                      validLib        = args['validLib'],
                      noTime          = args['noTime'],
                      kdWorkers       = args['kdWorkers'],
                      ignoreNan       = args['ignoreNan'] )

    err = ComputeError( df['Observations'], df['Predictions'] )
    return err['rho']

#-----------------------------------------------------
# Function to evaluate Simplex in PredictInterval Pool
#-----------------------------------------------------
def PredictIntervalSimplexFunc( Tp, data, args ) :

    df = API.Simplex( dataFrame       = data,
                      columns         = args['columns'],
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = args['E'], 
                      Tp              = Tp,
                      tau             = args['tau'],
                      exclusionRadius = args['exclusionRadius'],
                      embedded        = args['embedded'],
                      validLib        = args['validLib'],
                      noTime          = args['noTime'],
                      kdWorkers       = args['kdWorkers'],
                      ignoreNan       = args['ignoreNan'] )

    err = ComputeError( df['Observations'], df['Predictions'] )
    return err['rho']

#------------------------------------------------------------
# Function to evaluate Simplex in PredictExclusionRadius Pool
#------------------------------------------------------------
def PredictExclusionRadiusSimplexFunc( exclusionRadius, data, args ) :

    df = API.Simplex( dataFrame       = data,
                      columns         = args['columns'],
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = args['E'], 
                      Tp              = args['Tp'],
                      tau             = args['tau'],
                      exclusionRadius = exclusionRadius,
                      embedded        = args['embedded'],
                      validLib        = args['validLib'],
                      noTime          = args['noTime'],
                      kdWorkers       = args['kdWorkers'],
                      ignoreNan       = args['ignoreNan'] )

    err = ComputeError( df['Observations'], df['Predictions'] )
    return err['rho']

#-----------------------------------------------------
# Function to evaluate SMap in PredictNonlinear Pool
#-----------------------------------------------------
def PredictNLSMapFunc( theta, data, args ) :

    S = API.SMap( dataFrame       = data,
                  columns         = args['columns'],
                  target          = args['target'], 
                  lib             = args['lib'],
                  pred            = args['pred'],
                  E               = args['E'], 
                  Tp              = args['Tp'],
                  knn             = args['knn'],
                  tau             = args['tau'],
                  theta           = theta,
                  exclusionRadius = args['exclusionRadius'],
                  solver          = args['solver'],
                  embedded        = args['embedded'],
                  validLib        = args['validLib'],
                  noTime          = args['noTime'],
                  kdWorkers       = args['kdWorkers'],
                  ignoreNan       = args['ignoreNan'] )

    df = S['predictions']
    err = ComputeError( df['Observations'], df['Predictions'] )
    return err['rho']

#---------------------------------------------------------------------
# Single-point cross-map evaluation, reused by the E x Tp sweep below
#---------------------------------------------------------------------
def MVCM_ComputeXMapPair( dataFrame, columns, target, lib, pred, E, tau, Tp,
                          exclusionRadius, method, theta, embedded, noTime,
                          validEmbed=None ):
    """Evaluate one (columns, target, E, tau, Tp) cross-map.
       If validEmbed is supplied (precomputed for this E, tau), it's reused
       instead of recomputing Embed() -- saves redundant work when this is
       called repeatedly across a Tp sweep at fixed E."""

    if validEmbed is None:
        if embedded == False:
            embedding  = API.Embed( dataFrame=dataFrame, columns=columns, E=E, tau=tau )
        else:
            embedding = dataFrame[columns]
        validEmbed = embedding.notna().all( axis=1 )

    validTarget = dataFrame[target].shift( -Tp ).notna()
    validLib    = validEmbed & validTarget

    pointsRemoved = ( 2 * exclusionRadius ) + 1
    if validLib.sum() <= ( E + 1 ) + pointsRemoved :
        return tuple(columns), None, None

    # Base kwargs common to both Simplex and SMap
    kwargs = dict( dataFrame=dataFrame, columns=columns, target=target, lib=lib, pred=pred,
                   E=E, tau=tau, Tp=Tp, exclusionRadius=exclusionRadius, noTime=noTime,
                   embedded=embedded )

    if method == 'Simplex' :
        preds = API.Simplex( **kwargs, validLib=validLib )
    elif method == 'SMap' :
        preds = API.SMap( **kwargs, theta=theta )['predictions']

    # Record where successful predictions were made
    pred_mask = pd.DataFrame({ "Time": preds['Time'], "Mask": preds['Predictions'].notna() })

    return tuple(columns), pred_mask, preds

def _NanRecord( E, Tp, tau, metric, testSig, method, theta=None ) :
    rec = { 'E': E, 'Tp': Tp, 'tau': tau }
    if method == 'SMap':
        rec['theta'] = theta
    rec.update({metric: np.nan, 'N': 0})
    if testSig :
        rec['pval'] = np.nan
    return rec

#---------------------------------------------------------------------
# Worker: E x Tp grid scan for one (predictorCol -> targetCol) pair.
# Embeds predictorCol, predicts targetCol.
#---------------------------------------------------------------------
def MVCM_EmbedDimension( predictorCol, targetCol, dataFrame, args ) :
    """Scans E=1..maxE, Tp in tpGrid, for one column -> target pair.
       Returns (predictorCol, targetCol, dimTable), one row per (E,Tp)."""

    lib, pred       = args['lib'], args['pred']
    tau             = args['lagTau']        # embedding spacing = lagTau
    searchGrid      = args['searchGrid']    # Dict: {E: [valid Tp list]}
    testSig         = args['testSig']
    minN            = args.get('minN', 3)  # Default to 3 if not specified
    metric          = args['metric']
    exclusionRadius = args['exclusionRadius']
    noTime          = args['noTime']
    method          = args['method']
    theta           = args['theta']

    ### FIXME / TODO : add other metrics
    if metric == 'pearsonr':
        metricFunc = pearsonr
    elif metric == 'spearmanr':
        metricFunc = spearmanr
    else:
        raise RuntimeError(f'Need to add support for other metrics inside MVCM_EmbedDimension')

    records    = []
    for E in range( 1, 1000 ):   # prevent infinite loop
        if E not in searchGrid:
            break

        # Compute validEmbed once per E
        embedding  = API.Embed( dataFrame=dataFrame, columns=predictorCol, E=E, tau=tau )
        validEmbed = embedding.notna().all( axis=1 )

        # Loop over valid Tp values for this E
        for Tp in searchGrid[E]:
            _, _, preds = MVCM_ComputeXMapPair(
                dataFrame=dataFrame, columns=predictorCol, target=targetCol,
                lib=lib, pred=pred, E=E, tau=tau, Tp=Tp,
                exclusionRadius=exclusionRadius, method=method, theta=theta,
                embedded=False, noTime=noTime, validEmbed=validEmbed
            )

            if preds is None :
                records.append( _NanRecord( E, Tp, tau, metric, testSig, method=method, theta=theta ) )
                continue

            clean = preds[['Observations', 'Predictions']].dropna()
            if len( clean ) < minN :
                records.append( _NanRecord( E, Tp, tau, metric, testSig, method=method, theta=theta ) )
                continue

            rho, pval = metricFunc( clean['Observations'], clean['Predictions'] )
            rec = { 'E': E, 'Tp': Tp, 'tau': tau }
            if method == 'SMap':
                rec['theta'] = theta
            rec.update({metric: rho, 'N': len( clean )})
            if testSig :
                rec['pval'] = pval
            records.append( rec )

    return predictorCol, targetCol, pd.DataFrame( records )
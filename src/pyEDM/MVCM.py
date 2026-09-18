'''MVCM : Multi-View Cross Map gap filling.

   Fills missing values in a target time series using an ensemble of
   cross-map predictions from causally related columns in the same
   dataFrame (Carpenter et al. 2025).

   Pipeline:

     1. FindCausalPartners() - screen every candidate column against
        target with a Simplex E=1..maxE scan (Spearman rho + p-value),
        keep the nPartners columns with the strongest significant skill.

     2. BuildBlock()         - construct a symmetric (t-0, t-lag, t+lag)
        lagged embedding block from the causal partners.

     3. GenerateEmbeddings() - enumerate the small (size 2-3, or as given
        by comboSizes) column combinations from the block that are
        actually available (non-nan) at each gap row.

     4. CrossMap()           - cross-map target from every embedding
        found in step 3 via Simplex or SMap.

     5. FillGaps()           - fill each gap with a skill (rho) weighted
        average of the top-k embeddings that produced a valid prediction
        at that row.

   Optimize() runs a small theta/k grid search, scoring candidates by
   agreement with the *known* (non-gap) target values, and sets
   self.theta / self.k to the best combination it finds.
'''

# python modules
from itertools       import combinations, repeat
from multiprocessing import get_context
from warnings        import warn

# package modules
from numpy  import arange, argpartition, array, clip, divide, full, inf, isnan
from numpy  import maximum, nan, tile, where, zeros, zeros_like
from pandas import DataFrame, Series, concat, to_numeric, option_context
from pandas.api.types import is_integer_dtype
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# local modules
from .AuxFunc import IsIterable

#import pyEDM.API      as API
import pyEDM.PoolFunc  as PoolFunc

def _validate_shared( self ) :
    '''Shared validation logic for CrossMapScreener and MVCM.'''
    
    if self.Data is None :
        raise ValueError( f'{self.name} Validate(): dataFrame required.' )
    if not isinstance( self.Data, DataFrame ) :
        raise ValueError( f'{self.name} Validate(): dataFrame is not a Pandas DataFrame.' )
    
    firstCol = self.Data.columns[0]
    if firstCol != 'Time' :
        raise ValueError( f'{self.name} Validate(): First column in dataFrame must be named "Time", '
                          f'found "{firstCol}".' )
    timeVals = self.Data['Time']
    if not is_integer_dtype( timeVals ):
        raise TypeError( f'{self.name} Validate(): "Time" column must contain integer values.' )
    timeDiffs = timeVals.diff().dropna()
    if timeDiffs.empty:
        raise ValueError( f'{self.name} Validate(): "Time" column must contain at least 2 rows.' )
    step = int( timeDiffs.iloc[0] )
    if step == 0 :
        raise ValueError( f'{self.name} Validate(): "Time" column step size cannot be 0.' )
    if not (timeDiffs == step).all():
        raise ValueError( f'{self.name} Validate(): "Time" column must consist of evenly-spaced '
                            f'integers (found non-constant step sizes).' )
    self.timeStep = int( step )

    if not self.columns :
        raise ValueError( f'{self.name} Validate(): columns required.' )
    if not isinstance( self.columns, list ) :
        raise ValueError( f'{self.name} Validate(): columns must be a list.' )
    for column in self.columns :
        if not isinstance( column, str ) :
            raise ValueError( f'{self.name} Validate(): each element of columns must be a string, '
                              f'found {type(column).__name__}.' )
        if column not in self.Data.columns :
            raise ValueError( f'{self.name} Validate(): column {column} not found in dataFrame.' )
    if self.target in self.columns :
        raise ValueError( f'{self.name} Validate(): target should not be included '
                          f'in the list of columns for cross-mapping.' )

    if not self.target :
        raise RuntimeError( f'{self.name} Validate(): target required.' )
    if self.target not in self.Data.columns :
        raise ValueError( f'{self.name} Validate(): target {self.target} not found in dataFrame.' )
    if isinstance( self.target, (list, tuple) ) :
        if len( self.target ) == 1 :
            self.target = self.target[0]
        else :
            raise ValueError( f'{self.name} Validate(): target must be a single column name.' )
    if not isinstance( self.target, str ):
            raise ValueError( f'{self.name} Validate(): target must be a column name string.' )
    if self.target in self.columns :
        raise ValueError( f'{self.name} Validate(): target should not be included in the list of columns when cross-mapping.' )

    if not len( self.lib ) :
        raise ValueError( f'{self.name} Validate(): lib required.' )
    if not IsIterable( self.lib ) :
        self.lib = [ int(i) for i in self.lib.split() ]
    
    if not len( self.pred ) :
        raise ValueError( f'{self.name} Validate(): pred required.' )
    if not IsIterable( self.pred ) :
        self.pred = [ int(i) for i in self.pred.split() ]

    ####### FIXME / TODO : Add validation for testSig, alpha, minN, metric.
    
    if self.method not in ( 'Simplex', 'SMap' ) :
        raise ValueError( f'{self.name} Validate(): method must be "Simplex" or "SMap".' )
    if self.method == 'SMap' and self.theta is None :
        raise ValueError( f'{self.name} Validate(): theta is required when method = "SMap".' )
    
    if self.metric not in ( 'pearsonr', 'spearmanr' ) :
        raise ValueError( f'{self.name} Validate(): metric must be "pearsonr" or "spearmanr".' )

class _MakeBlock :
    def MakeBlock( self, partners_only, verbose=None ):
        '''Build an embedding block of the causal partner columns,
            expanding rows at both ends of the time axis if specified.
            If full = True, makes full block using all columns.
            Sets self.Block.'''
        is_verbose = self.verbose if verbose is None else verbose
        if is_verbose:
            print( f'{self.name}: MakeBlock()' )

        if self.Partners is None :
            raise RuntimeError( f'{self.name} BuildBlock(): call '
                                    'FindCausalPartners() first.')

        # Include lags of the target for gap filling
        if self.useTargetLags:
            maxLag = self.numLags * abs(self.lagTau)
            target_lags = [ -i * self.lagTau for i in range( -self.numLags, self.numLags + 1 ) ]
        else:
            target_lags = [0]

        # Collect unique lags across all partners
        if partners_only : 
            partner_lags = [lag for lags in self.Partners['blockLags'] for lag in lags]
        else :
            partner_lags = [ abs( self.numLags * self.lagTau ) ]
        all_lags = target_lags + partner_lags
        max_lag = max([abs(lag) for lag in all_lags]) if (self.expandTime and all_lags) else 0

        # Build base DataFrame (with expanded time index if expandTime=True)
        if self.expandTime and max_lag > 0 :
            N = len( self.Data )

            # Generate expanded Time column
            t_start = self.Data['Time'].iloc[0] - max_lag * self.timeStep
            t_end   = self.Data['Time'].iloc[-1] + max_lag * self.timeStep
            expanded_time = arange( t_start, t_end + self.timeStep, self.timeStep, dtype=int )

            padded_dict = { 'Time': expanded_time }
            for col in self.Data.columns:
                if col != 'Time':
                    padded = full( len(expanded_time), nan )
                    padded[ max_lag : max_lag + N ] = self.Data[col].values
                    padded_dict[col] = padded

            data_expanded = DataFrame( padded_dict )
        else:
            data_expanded = self.Data.copy()

        # Identify gap time points in target on the (possibly expanded) time axis
        self.gapTimes = data_expanded.loc[ data_expanded[self.target].isna(), 'Time' ].to_numpy()

        # Build lagged feature block
        selected_cols = []
        block_dict    = { 'Time': data_expanded['Time'] }

        # Add target lags
        for lag in target_lags:
            col_name = f"{self.target}(t{lag if lag < 0 else f'+{lag}' if lag > 0 else '-0'})"
            selected_cols.append( col_name )
            block_dict[col_name] = data_expanded[self.target].shift( -1 * lag )
        
        # Add column lags
        if partners_only :
            for _, row in self.Partners.iterrows():
                col  = row['causalPartner']
                lags = row['blockLags']

                for lag in lags:
                    col_name = f"{col}(t{lag if lag < 0 else f'+{lag}' if lag > 0 else '-0'})"
                    selected_cols.append( col_name )
                    block_dict[col_name] = data_expanded[col].shift( -1 * lag )
            self.Block = DataFrame( block_dict )[ ['Time'] + selected_cols ]

            return self.Block
        else :
            for col in self.columns :
                lags = [ -i * self.lagTau for i in range( -self.numLags, self.numLags + 1 ) ]
                
                for lag in lags :
                    col_name = f"{col}(t{lag if lag < 0 else f'+{lag}' if lag > 0 else '-0'})"
                    selected_cols.append( col_name )
                    block_dict[col_name] = data_expanded[col].shift( -1 * lag )
            self.FullBlock = DataFrame( block_dict )[ ['Time'] + selected_cols ]

            return self.FullBlock

#--------------------------------------------------------------------
# CrossMapScreener: screens candidate time series for cross-map skill
#--------------------------------------------------------------------
class CrossMapScreener( _MakeBlock ) :
    '''Screens candidate time series against a target variable 
       for cross-map skill and identifies relevant lags.
    '''

    def __init__( self,
                  dataFrame        = None,
                  columns          = "",
                  target           = "",
                  lib              = "",
                  pred             = "",
                  method           = 'Simplex',
                  theta            = None,
                  numLags          = None,
                  lagTau           = None,
                  metric           = 'pearsonr',
                  minN             = 0,
                  metricCutoff     = None,
                  testSig          = False,
                  alpha            = None,
                  nPartners        = None,
                  useTargetLags    = True,
                  expandTime       = False,
                  verbose          = False,
                  numProcess       = 4,
                  returnObject     = False,
                  showPlot         = False,
                  ):
        '''Initialize CrossMapScreener.'''

        # Assign parameters from API arguments
        self.name            = 'CrossMapScreener'
        self.Data            = dataFrame
        self.columns         = columns
        self.target          = target
        self.lib             = lib
        self.pred            = pred
        self.method          = method
        self.theta           = theta
        self.numLags         = numLags
        self.lagTau          = lagTau
        self.metric          = metric
        self.minN            = minN
        self.metricCutoff    = metricCutoff
        self.testSig         = testSig
        self.alpha           = alpha
        self.nPartners       = nPartners
        self.useTargetLags   = useTargetLags
        self.expandTime      = expandTime
        self.verbose         = verbose
        self.numProcess      = numProcess
        self.returnObject    = returnObject
        self.showPlot        = showPlot

        # FIXME / TODO : Internal default parameters (under testing)
        self.E               = 1        # Screens over a range of E's implicitly defined by numLags and lagTau
        self.Tp              = 0        # Screens over a range of Tps implicitly defined by numLags and lagTau
        self.tau             = -1       # tau is equivalent to lagTau -- maybe I should just call lagTau = tau?
        self.exclusionRadius = 0        # Have not tested with exclusionRadius > 0
        self.noTime          = False    # Have not tested with noTime = True
        self.embedded        = False    # Input should not be pre-embedded
        self.ignoreNan       = True     # Have not tested with ignoreNan = False
        self.generateSteps   = 0        # Have not tested with generateSteps > 0. Is this equivalent to expandTime?
        self.generateConcat  = False    # Have not tested with generateConcat = True
        self.mpMethod        = None     # Have not tested with other mpMethod
        self.chunksize       = 1        # Have not tested with other chunksize
        # self.selfWeight    = None     # Reserved for future noise filtering

        # Outputs, populated by Run()
        self.Partners        = None
        self.Block           = None
        self.gapTimes        = None

        # Setup
        self.Validate()

        # 1st data column is time
        self.time = self.Data.iloc[ :, 0 ].to_numpy()

    def Validate( self ):
        '''Custom CrossMapScreener Validation.'''
        if self.verbose :
            print( f'{self.name} Validate():' )

        _validate_shared( self )

        if not self.numLags :
            raise RuntimeError( f'{self.name} Validate(): numLags required.' )
        if not isinstance( self.numLags, int ) : 
            raise TypeError( f'{self.name} Validate(): numLags must be an integer.' )
        if not self.lagTau :
            raise RuntimeError( f'{self.name} Validate(): lagTau required.' )
        if not isinstance( self.lagTau, int ) : 
            raise TypeError( f'{self.name} Validate(): lagTau must be an integer.' )
        
        if not self.nPartners :
            self.nPartners = len( self.columns )
    
    def PlotGapAvailability( self, full_block, gap_times ):
        '''Visualize target series alongside available predictor counts in 
           the unfiltered full block vs. filtered causal subset.'''
        # Feature columns exclude 'Time'
        full_feature_cols   = [ col for col in full_block.columns if col != 'Time' ]
        causal_feature_cols = [ col for col in self.Block.columns if col != 'Time' ]

        # Count non-NaN values across available features for each time step
        n_full   = full_block[full_feature_cols].notna().sum( axis=1 )
        n_subset = self.Block[causal_feature_cols].notna().sum( axis=1 )

        fig, ax = plt.subplots( figsize=(10, 3.5) )

        # Plot target observation series
        ax.plot( self.Data['Time'], self.Data[self.target], marker='.', color='black', label=f'{self.target} (Target)' )
        ax.set_title( f'{self.target}\nand Number of Features Available for Gap Filling' )
        ax.set_ylabel( 'Target Value' )
        ax.set_xlabel( 'Time' )

        # Overlay secondary Y-axis with availability bars
        ax2 = ax.twinx()
        ax2.bar( full_block['Time'], n_full, alpha=0.35, color='gray', label='All Candidate Lags (Full)' )
        ax2.bar( self.Block['Time'], n_subset, alpha=0.65, color='tab:blue', label='Selected Features' )
        ax2.set_ylabel( 'Number of Available Features' )

        # Combine legends across twin axes
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend( lines_1 + lines_2, labels_1 + labels_2, loc='upper right' )

        plt.tight_layout()
        plt.show()

        # Display availability breakdown across missing target points
        gap_mask = self.Block['Time'].isin( gap_times )
        n_subset_df = DataFrame({
            'Time': self.Block.loc[gap_mask, 'Time'],
            'All Available Features': n_full[gap_mask],
            'Selected Features': n_subset[gap_mask]
        }).set_index('Time')

        if self.verbose :
            print( 'Feature availability at missing target rows:' )
            with option_context( 'display.max_rows', None ):
                display( n_subset_df )

    def Run( self ) :
        '''Screens self.columns against target across E/Tp grid relevant for block construction.
           Returns DataFrame of selected causal partners and their required lags.
        '''

        # Construct E/Tp searchGrid based on numLags and lagTau
        maxLag = self.numLags * abs( self.lagTau )
        tpRange = [ i * self.lagTau for i in range( -self.numLags, self.numLags + 1 ) ]
        searchGrid = {}
        for E in range( 1, 2 * self.numLags + 2 ) :   # Maximum possible E across the lag window [-nτ, nτ]
            validTp = [ Tp for Tp in tpRange 
                        if all( -maxLag <= ( Tp - k * self.lagTau ) <= maxLag 
                                for k in range(E) ) ]
            if not validTp:
                break
            searchGrid[ E ] = validTp

        args = { 'lib'            : self.lib,
                 'pred'            : self.pred,
                 'lagTau'          : self.lagTau,
                 'searchGrid'      : searchGrid,
                 'testSig'         : self.testSig,
                 'minN'            : self.minN,
                 'metric'          : self.metric,
                 'exclusionRadius' : self.exclusionRadius,
                 'noTime'          : self.noTime,
                 'method'          : self.method,
                 'theta'           : self.theta }

        # Build tasks
        poolArgs = [ ( col, self.target, self.Data, args ) for col in self.columns ]

        mpContext = get_context( self.mpMethod )
        with mpContext.Pool( processes = self.numProcess ) as pool :
            results = pool.starmap( PoolFunc.MVCM_EmbedDimension, poolArgs,
                                    chunksize = getattr( self, 'chunksize', 1 ) )

        rows = []
        for partner, target, dimTable in results :
            sig = dimTable.dropna( subset = [self.metric] )
            sig = sig[ sig['N'] >= self.minN ]      # Enforce minN if specified (default 0)
            if self.metricCutoff:
                sig = sig[ sig[self.metric] > self.metricCutoff ] ### FIXME / TODO : You may want to filter for metric < cutoff for other metrics
            if self.testSig :
                # Filter for statistically significant cross-maps
                sig = sig[ sig['pval'] < self.alpha ]
            if sig.empty :
                continue

            # Extract union of all lags across all signficant embeddings
            union_lags = set()
            for _, row in sig.iterrows():
                e_val = int( row['E'] )
                tp_val = int( row['Tp'] )
                # Reindex embedding lags (relative to library point t) into lags
                # relative to the target row t' = t + Tp being predicted:
                # col(t' - lag_k), lag_k = -Tp + k * lagTau, k = 0..E-1
                e_lags = [ -tp_val + k * self.lagTau for k in range( e_val ) ]
                union_lags.update( e_lags )

            # Best embedding stats (retained for metadata and ranking)
            best   = sig.loc[ sig[self.metric].idxmax() ]
            bestE  = int( best['E'] )
            bestTp = int( best['Tp'] )

            row = { 'causalPartner'       : partner,
                    'blockLags'           : sorted( list( union_lags ) ),
                    'bestUnivariate'      : f'E={bestE}, τ={self.lagTau}, Tp={bestTp}',
                    'theta'               : None if self.method == 'Simplex' else self.theta,
                    f'best_{self.metric}' : float( best[self.metric] ),
                    'pval'                : None if not self.testSig else float( best['pval'] ),
                    'N'                   : int( best['N'] ) }
            if self.method == 'Simplex' :
                row.pop( 'theta', None )
            if self.testSig == False :
                row.pop( 'pval', None )

            rows.append( row )

        if not rows :
            raise RuntimeError( f'{self.name} FindCausalPartners(): No column '
                                f'significantly cross-maps {self.target}. ')

        self.Partners = ( DataFrame( rows )
                        .sort_values( f'best_{self.metric}', ascending = False )
                        .reset_index( drop = True )
                        .head( self.nPartners ) )
        
        self.MakeBlock( partners_only=True )
        
        # Plot number of columns available to fill each gap
        if self.showPlot and len( self.gapTimes ) > 0:
            full_block = self.MakeBlock( partners_only=False, verbose=False )
            self.PlotGapAvailability( full_block=full_block, gap_times=self.gapTimes )

        return self.Partners

#------------------------------------------------------------------
class MVCM( _MakeBlock ) :
    '''MVCM class.'''

    def __init__( self,
                  dataFrame        = None,
                  partners         = None,
                  columns          = "",
                  target           = "",
                  lib              = "",
                  pred             = "",
                  method           = 'Simplex',
                  theta            = None,
                  numLags          = None,
                  lagTau           = None,
                  D                = None,
                  k                = None,
                  metric           = 'pearsonr',
                  minN             = 0,
                  metricCutoff     = None,
                  testSig          = True,
                  alpha            = 0.05,
                  nPartners        = None,
                  useTargetLags    = True,
                  expandTime       = False,
                  verbose          = False,
                  numProcess       = 4,
                  returnObject     = False,
                  showPlot         = False,
                  ):
        '''Initialize MVCM.'''

        # Assign parameters from API arguments
        self.name            = 'MVCM'
        self.Data            = dataFrame
        self.Partners        = partners
        self.columns         = columns
        self.target          = target
        self.lib             = lib
        self.pred            = pred
        self.method          = method
        self.theta           = theta
        self.numLags         = numLags
        self.lagTau          = lagTau
        self.D               = D
        self.k               = k
        self.metric          = metric
        self.minN            = minN
        self.metricCutoff    = metricCutoff
        self.testSig         = testSig
        self.alpha           = alpha
        self.nPartners       = nPartners
        self.useTargetLags   = useTargetLags
        self.expandTime      = expandTime
        self.verbose         = verbose
        self.numProcess      = numProcess
        self.returnObject    = returnObject
        self.showPlot        = showPlot 

        # FIXME / TODO : Internal default parameters (under testing)
        self.E               = 1
        self.Tp              = 0
        self.tau             = -1
        self.exclusionRadius = 0
        self.noTime          = False
        self.embedded        = False
        self.ignoreNan       = True
        self.generateSteps   = 0
        self.generateConcat  = False
        self.mpMethod        = None
        self.chunksize       = 1
        self.selfWeight      = None # Reserved for future noise filtering

        # # Outputs, populated by Run() / individual stage methods
        # self.Embeddings        = None  # list of column-name tuples
        # self.EmbeddingResults  = None  # dict: embedding -> {'rho','predictions'}
        # self.EmbeddingSummary  = None  # DataFrame view of EmbeddingResults
        # self.Filled            = None  # DataFrame: Time,Observations,Predictions,Filled
        # self._ensembleCache    = None  # cached matrices for FillGaps/Optimize

        # Setup
        self.Validate()

        # 1st data column is time
        self.time = self.Data.iloc[ :, 0 ].to_numpy()

        # # Identify gap time points in target
        # self.gapTimes = self.Data.loc[ self.Data[self.target].isna(), 'Time' ].to_numpy()

    #--------------------------------------------------------------------
    def Validate( self ):
    #--------------------------------------------------------------------
        '''Custom MVCM validation.'''
        if self.verbose :
            print( f'{self.name}: Validate()' )

        _validate_shared( self )

        # Only require screener-related params if Partners is not given.
        if self.Partners is None :
            if not self.numLags :
                raise RuntimeError( f'{self.name} Validate(): numLags required.' )
            if not isinstance( self.numLags, int ) : 
                raise TypeError( f'{self.name} Validate(): numLags must be an integer.' )
            if not self.lagTau :
                raise RuntimeError( f'{self.name} Validate(): lagTau required.' )
            if not isinstance( self.lagTau, int ) : 
                raise TypeError( f'{self.name} Validate(): lagTau must be an integer.' )

        if not self.D :
            raise RuntimeError( f"Validate() {self.name}: D required." )
        if not self.k :
            raise RuntimeError( f"Validate() {self.name}: k required." )
    #--------------------------------------------------------------------
    def GenerateEmbeddings( self ):
    #--------------------------------------------------------------------
        '''Enumerate every partner-lag column combination (sized per
           self.D) that is fully available (non-nan) at each gap
           row of the target. Sets self.Embeddings to the union across
           all gap rows: the minimal set that needs to be cross-mapped
           to have a chance of filling every gap.'''
        if self.verbose :
            print(f'{self.name}: GenerateEmbeddings()')
        
        if self.Block is None :
            raise RuntimeError( f'{self.name} GenerateEmbeddings(): call '
                                'BuildBlock() first.' )
        
        targetStr     = f"{self.target}(t-0)"
        exclude       = {targetStr, "Time"}
        candidateCols = [ c for c in self.Block.columns if c not in exclude ]

        gapRows   = self.Block.index[ self.Block[ targetStr ].isna() ]
        validMask = self.Block.loc[ gapRows, candidateCols ].notna()

        embeddingSet = set()
        for row in gapRows :
            available = [ c for c in candidateCols if validMask.loc[row, c] ]
            if len( available ) >= self.D :
                embeddingSet.update( combinations( available, self.D ) )
            
        self.Embeddings = sorted( embeddingSet )

        if not len( self.Embeddings ) :
            raise RuntimeError( f'{self.name} GenerateEmbeddings(): no '
                                'embedding is available at '
                                'any gap row. Increase nPartners/numLags, '
                                'or reduce alpha/minN.' )
        return self.Embeddings
    
    #--------------------------------------------------------------------
    def CrossMap( self, theta = None ):
    #--------------------------------------------------------------------
        '''Cross-map target from every embedding in self.Embeddings via
           Simplex or SMap. Sets self.EmbeddingResults / EmbeddingSummary.
        '''
        if self.verbose :
            print( f'{self.name}: CrossMap()' )

        if self.Embeddings is None :
            raise RuntimeError( f'{self.name} CrossMap(): call '
                                'GenerateEmbeddings() first.' )
        
        ### FIXME / TODO : add other metrics
        if self.metric == 'pearsonr':
            metricFunc = pearsonr
        elif self.metric == 'spearmanr':
            metricFunc = spearmanr
        else:
            raise RuntimeError(f'Need to add support for other metrics inside MVCM_EmbedDimension')
        
        theta  = self.theta if theta is None else theta
        N      = self.Block.shape[0]
        libStr = predStr = f'1 {N}'

        poolTasks = [
            (
                self.Block,               # dataFrame
                list( embedCols ),        # columns
                f"{self.target}(t-0)",    # target
                libStr,                   # lib
                predStr,                  # pred
                len( embedCols ),         # E = dimension D
                self.lagTau,              # tau
                0,                        # Tp
                self.exclusionRadius,     # exclusionRadius
                self.method,              # method
                theta,                    # theta
                True,                     # embedded
                False,                    # noTime
                None                      # validEmbed
            )
            for embedCols in self.Embeddings
        ]

        mpContext = get_context( self.mpMethod )
        chunksize = getattr( self, 'chunksize', 1 )

        with mpContext.Pool( processes = self.numProcess ) as pool :
            results = pool.starmap( PoolFunc.MVCM_ComputeXMapPair, poolTasks, chunksize=chunksize )

        embeddingResults = {}
        rows = []
        for embedding, _, preds in results :
            if preds is not None :
                clean = preds[ ['Observations','Predictions'] ].dropna()
                if len( clean ) < self.minN :
                    continue
                rho, pval = metricFunc( clean['Observations'], clean['Predictions'] )
            else :
                rho = None

            embeddingResults[ embedding ] = { self.metric : rho, 'predictions' : preds }

            row =  { 'embedding' : embedding,
                     'E'         : len( embedding ),
                     'theta'     : None if self.method == 'Simplex' else self.theta,
                     self.metric : rho,
                     'pval'      : pval,
                     'N'         : int( preds['Predictions'].notna().sum() )
                                    if preds is not None else 0 }
            if self.method == 'Simplex' :
                row.pop( 'theta', None )
            if self.testSig == False :
                row.pop( 'pval', None )
            
            rows.append( row )

        self.EmbeddingResults = embeddingResults
        self.EmbeddingSummary = ( DataFrame( rows )
                                  .sort_values( self.metric, ascending = False )
                                  .reset_index( drop = True ) )
        self._ensembleCache = None   # invalidate cache

        return self.EmbeddingSummary

    #--------------------------------------------------------------------
    def _PrepareEnsemble( self ):
    #--------------------------------------------------------------------
        '''Build (and cache) predMatrix / rhoMatrix / maskMatrix used by
           FillGaps(), aligned to self.Block rows via the Time column 
           each Simplex/SMap result carries.'''
        if self._ensembleCache is not None :
            return self._ensembleCache

        if self.EmbeddingResults is None :
            raise RuntimeError( f'{self.name}: call CrossMap() first.' )

        validEmbs = [ e for e, r in self.EmbeddingResults.items()
                      if r['predictions'] is not None and r[self.metric] is not None ]
        if not len( validEmbs ) :
            raise RuntimeError( f'{self.name}: no embedding produced a '
                                'valid cross-map prediction.' )

        rhos = array( [ self.EmbeddingResults[e][self.metric] for e in validEmbs ],
                      dtype = float )
        
        N = len( self.Block )

        predMatrix = zeros( (N, len(validEmbs)) )
        maskMatrix = zeros( (N, len(validEmbs)), dtype = bool )

        for j, e in enumerate( validEmbs ) :
            preds   = self.EmbeddingResults[e]['predictions'][['Time','Predictions']]
            aligned = DataFrame( { 'Time': self.Block["Time"] } ) \
                        .merge( preds, on = 'Time', how = 'left' )
            predMatrix[:, j] = aligned['Predictions'].to_numpy()
            maskMatrix[:, j] = aligned['Predictions'].notna().to_numpy()

        invalid = ~maskMatrix | isnan( predMatrix )
        predMatrix[ invalid ] = 0.0

        rhoMatrix = tile( rhos, (N, 1) ).astype( float )
        rhoMatrix[ invalid ] = -inf

        self._ensembleCache = dict( validEmbs  = validEmbs,
                                    predMatrix = predMatrix,
                                    rhoMatrix  = rhoMatrix,
                                    invalid    = invalid )
        
        return self._ensembleCache

    #--------------------------------------------------------------------
    def _EstimateAtK( self, k, selfWeight ):
    #--------------------------------------------------------------------
        '''Skill-weighted top-k ensemble estimate at every block row.
           Vectorized re-implementation of the notebook's
           optimized_fill_gaps()/optimize_k() inner loop.'''
        cache      = self._PrepareEnsemble()
        predMatrix = cache['predMatrix']
        rhoMatrix  = cache['rhoMatrix']
        invalid    = cache['invalid']

        N, M = rhoMatrix.shape
        k = min( max( k, 1 ), M )

        topK_i = argpartition( -rhoMatrix, k - 1, axis = 1 )[ :, :k ]

        finalMask = zeros_like( rhoMatrix, dtype = bool )
        rows      = arange( N )[ :, None ]
        finalMask[ rows, topK_i ] = True
        finalMask[ invalid ]      = False

        weightedRhos = where( finalMask, rhoMatrix, 0.0 )
        weightedRhos = clip( weightedRhos, 0, None )

        rowSums = weightedRhos.sum( axis = 1, keepdims = True )
        weights = zeros_like( weightedRhos )
        divide( weightedRhos, rowSums, out = weights, where = rowSums != 0 )

        if selfWeight is None:
            selfWeight = 0.0
        weights *= ( 1 - selfWeight / 100 )

        estimates = ( predMatrix * weights ).sum( axis = 1 )
        estimates = maximum( estimates, 0 )
        return estimates

    #--------------------------------------------------------------------
    def FillGaps( self, k = None, selfWeight = None ):
    #--------------------------------------------------------------------
        '''Fill target gaps with the top-k skill-weighted ensemble
           estimate. Sets self.Filled (Time, Observations, Predictions,
           Filled), trimmed back to the original (non-padded) time range.
        '''
        if self.verbose :
            print( f'{self.name}: FillGaps()' )

        k          = self.k          if k          is None else k
        selfWeight = self.selfWeight if selfWeight is None else selfWeight
        if not k :
            raise ValueError( f'{self.name} FillGaps(): k must be > 0. '
                              'Set MVCM(..., k = <int>) or call Optimize() '
                              'first.' )

        estimates = self._EstimateAtK( k, selfWeight )

        observations = self.Block[ f"{self.target}(t-0)" ].to_numpy()
        filled       = observations.copy()
        gapMask      = isnan( filled )
        filled[ gapMask ] = estimates[ gapMask ]

        out = DataFrame( { 'Time'         : self.Block[ "Time" ],
                           'Observations' : observations,
                           'Predictions'  : estimates,
                           'Filled'       : filled } )

        self.Filled     = out.reset_index( drop = True )
        self.Projection = self.Filled
        return self.Filled

    #--------------------------------------------------------------------
    def Run( self, optimize = False ):
    #--------------------------------------------------------------------
        '''Run the full pipeline in order. If optimize is True (or k/theta
           were left unresolved), run Optimize() before the final
           FillGaps(). Returns self.Filled.'''
        if self.verbose :
            print( f'{self.name}: Run()' )

        # Step 1: Only run CrossMapScreener if partners were not passed in
        if self.Partners is None:
            if self.verbose:
                print(f'{self.name}: No pre-computed partners found. Running CrossMapScreener...')
            _verbose = False
            _showPlot = False
            screener = CrossMapScreener( dataFrame=self.Data, columns=self.columns, target=self.target,
                                              lib=self.lib, pred=self.pred, method=self.method, theta=self.theta,
                                              numLags=self.numLags, lagTau=self.lagTau, metric=self.metric,
                                              minN=self.minN, metricCutoff=self.metricCutoff, testSig=self.testSig,
                                              alpha=self.alpha, nPartners=self.nPartners, 
                                              useTargetLags=self.useTargetLags, expandTime=self.expandTime, 
                                              verbose=_verbose, numProcess=self.numProcess, 
                                              returnObject=self.returnObject, showPlot=_showPlot)
            screener.Run()
            self.Partners = screener.Partners
            self.Block = screener.Block
        else:
            if self.verbose:
                print(f'{self.name}: Using pre-computed partners DataFrame.')

            ### FIXME / TODO : MakeBlock still requires numLags/lagTau, but the rest of MVCM doesn't need it
            #                  if Partners is given.
            self.MakeBlock( partners_only=True )

        self.GenerateEmbeddings()
        self.CrossMap()
        self.FillGaps()

        if self.showPlot :
            from matplotlib.pyplot import show
            ax = self.Filled.plot(
                x='Time',
                y=['Filled', 'Observations'],
                color=['C1', 'C0'],
                style='.-',
                figsize=(10, 3),
                title=f'{self.target} : MVCM gap filled'
            )
            ax2 = self.Filled.plot( x = 'Time', y = ['Observations', 'Predictions'],
                                   style = '.-', figsize = (10,3),
                                   title = f'{self.target} : Observations vs. Predictions' )
            show()

        return self.Filled

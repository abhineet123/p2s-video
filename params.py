

class Params:
    """
    :ivar alsologtostderr:  also log to stderr? (default: 'false')
    :type alsologtostderr: bool

    :ivar config:  The config file. (default: 'path/to/config/file.py')
    :type config: NoneType

    :ivar delta_threshold:  Log if history based diff crosses this threshold. (default: '0.5') (a number)
    :type delta_threshold: float

    :ivar dist:  Whether to use distributed strategy (default: 'false')
    :type dist: bool

    :ivar grain_num_threads_computing_num_records:  The number of threads used to fetch file instructions (i.e.,
    the max number of Array Record files opened while calculating the total number of records). (default: '64') (an
    integer)
    :type grain_num_threads_computing_num_records: int

    :ivar grain_num_threads_fetching_records:  The number of threads used to fetch records from Array Record files. (
    i.e., the max number of Array Record files opened while fetching records). (default: '64') (an integer)
    :type grain_num_threads_fetching_records: int

    :ivar hbm_oom_exit:  Exit the script when the TPU HBM is OOM. (default: 'true')
    :type hbm_oom_exit: bool

    :ivar log_dir:  directory to write logfiles into (default: '')
    :type log_dir: str

    :ivar logger_levels:  Specify log level of loggers. The format is a CSV list of `name:level`. Where `name` is the
    logger name used with `logging.getLogger()`, and `level` is a level name  (INFO, DEBUG, etc). e.g.
    `myapp.foo:INFO,other.logger:DEBUG` (default: '')
    :type logger_levels: dict

    :ivar logtostderr:  Should only log to stderr? (default: 'false')
    :type logtostderr: bool

    :ivar master:  Address/name of the TensorFlow master to use.
    :type master: NoneType

    :ivar mode:  <train|eval>: train or eval (default: 'train')
    :type mode: str

    :ivar model_dir:  Directory to store checkpoints and summaries.
    :type model_dir: NoneType

    :ivar only_check_args:
    :type only_check_args: bool

    :ivar op_conversion_fallback_to_while_loop:
    :type op_conversion_fallback_to_while_loop: bool

    :ivar pdb:  Alias for --pdb_post_mortem. (default: 'false')
    :type pdb: bool

    :ivar pdb_post_mortem:  Set to true to handle uncaught exceptions with PDB post mortem. (default: 'false')
    :type pdb_post_mortem: bool

    :ivar profile_file:  Dump profile information to a file (for python -m pstats). Implies --run_with_profiling.
    :type profile_file: NoneType

    :ivar run_eagerly:  Whether to run eagerly (for interactive debugging). (default: 'false')
    :type run_eagerly: bool

    :ivar run_with_pdb:  Set to true for PDB debug mode (default: 'false')
    :type run_with_pdb: bool

    :ivar run_with_profiling:  Set to true for profiling the script. Execution will be slower, and the output format
    might change over time. (default: 'false')
    :type run_with_profiling: bool

    :ivar runtime_oom_exit:  Exit the script when the TPU runtime is OOM. (default: 'true')
    :type runtime_oom_exit: bool

    :ivar showprefixforinfo:  If False, do not prepend prefix to info messages when it's logged to stderr,
    --verbosity is set to INFO level, and python logging is used. (default: 'true')
    :type showprefixforinfo: bool

    :ivar stderrthreshold:  log messages at this level, or more severe, to stderr in addition to the logfile.
    Possible values are 'debug', 'info', 'warning', 'error', and 'fatal'.  Obsoletes --alsologtostderr. Using
    --alsologtostderr cancels the effect of this flag. Please also note that this flag is subject to --verbosity and
    requires logfile not be stderr. (default: 'fatal') -v,--verbosity: Logging verbosity level. Messages logged at
    this level or lower will be included. Set to 1 for debug logging. If the flag was not set or supplied,
    the value will be changed from the default of -1 (warning) to 0 (info) after flags are parsed. (default: '-1') (
    an integer)
    :type stderrthreshold: str

    :ivar test_random_seed:  Random seed for testing. Some test frameworks may change the default value of this flag
    between runs, so it is not appropriate for seeding probabilistic tests. (default: '301') (an integer)
    :type test_random_seed: int

    :ivar test_randomize_ordering_seed:  If positive, use this as a seed to randomize the execution order for test
    cases. If "random", pick a random seed to use. If 0 or not set, do not randomize test case execution order. This
    flag also overrides the TEST_RANDOMIZE_ORDERING_SEED environment variable. (default: '')
    :type test_randomize_ordering_seed: str

    :ivar test_srcdir:  Root of directory tree where source files live (default: '')
    :type test_srcdir: str

    :ivar test_tmpdir:  Directory for temporary testing files (default: '/tmp/absl_testing')
    :type test_tmpdir: str

    :ivar tfds_debug_list_dir:  Debug the catalog generation (default: 'false')
    :type tfds_debug_list_dir: bool

    :ivar tfhub_cache_dir:  If set, TF-Hub will download and cache Modules into this directory. Otherwise it will
    attempt to find a network path.
    :type tfhub_cache_dir: NoneType

    :ivar tfhub_model_load_format:  <COMPRESSED|UNCOMPRESSED|AUTO>: If set to COMPRESSED, archived modules will be
    downloaded and extractedto the `TFHUB_CACHE_DIR` before being loaded. If set to UNCOMPRESSED, themodules will be
    read directly from their GCS storage location withoutneeding a cache dir. AUTO defaults to COMPRESSED behavior. (
    default: 'AUTO')
    :type tfhub_model_load_format: str

    :ivar tt_check_filter:  Terminate early to check op name filtering. (default: 'false')
    :type tt_check_filter: bool

    :ivar tt_single_core_summaries:  Report single core metric and avoid aggregation. (default: 'false')
    :type tt_single_core_summaries: bool

    :ivar use_cprofile_for_profiling:  Use cProfile instead of the profile module for profiling. This has no effect
    unless --run_with_profiling is set. (default: 'true')
    :type use_cprofile_for_profiling: bool

    :ivar use_tpu:  Whether to use tpu. (default: 'false')
    :type use_tpu: bool

    :ivar v:
    :type v: int

    :ivar verbosity:
    :type verbosity: int

    :ivar xml_output_file:  File to store XML test results (default: '')
    :type xml_output_file: str

    """

    def __init__(self):
        self.cfg = ()

        self.config = None
        self.dist = False
        self.master = None
        self.mode = 'train'
        self.model_dir = None
        self.run_eagerly = False
        self.use_tpu = False

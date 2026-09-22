"""
Python wrapper (via JPype) around LogMap's java-based `LogMapLLM_Interface`:

    https://github.com/ernestojimenezruiz/logmap-matcher/blob/master/src/main/java/uk/ac/ox/krr/logmap2/LogMapLLM_Interface.java

Originally ported from:

    https://github.com/jonathondilworth/logmap-llm/blob/jd-extended/logmap_interface.py

Usage notes:

- start_jvm() must have run (with logmap on the classpath) before constructing
  a LogMapInterface or importing logmap_llm.bridging.
- Ontology filepaths provided to the interface require a 'file:' URI prefix.
- The parameters directory path must end with `os.sep` (e.g., `/`).
- The output directory must already exist, and is mutable (this has
  implications when running multiple processes at once); set it before the
  initial alignment and again before refinement.

TODO: write tests — an end-to-end alignment run (download mappings, set up the
directory structure, align, verify output), plus setter round-trip checks for
params dir, output dir, and the extended M_ask flag (the getters require
LogMap src changes; see end of file).
"""

import os
import jpype                # type: ignore
import jpype.imports        # type: ignore
from pathlib import Path

from logmap_llm.config.schema import LogMapLLMConfig


def ontology_file_uri(path: str | Path) -> str:
    """Render a local ontology path as a standards-compliant escaped file URI."""
    return Path(path).expanduser().resolve().as_uri()


def start_jvm(logmap_dir: str | Path, max_heap: str = "8g") -> None:
    """Start the LogMap JVM after checking its external bundle layout."""
    if jpype.isJVMStarted():
        return

    logmap_root = Path(logmap_dir).expanduser().resolve()
    logmap_jar = logmap_root / "logmap-matcher-4.0.jar"
    dependencies = logmap_root / "java-dependencies"
    parameters = logmap_root / "parameters.txt"
    if not logmap_jar.is_file():
        raise FileNotFoundError(f"LogMap JAR not found: {logmap_jar}")
    if not dependencies.is_dir():
        raise FileNotFoundError(f"LogMap dependencies directory not found: {dependencies}")
    if not parameters.is_file():
        raise FileNotFoundError(f"LogMap parameters file not found: {parameters}")

    jpype.addClassPath(str(logmap_jar))
    jpype.addClassPath(str(dependencies / "*"))

    jpype.startJVM(
        f"-Xmx{max_heap}",
        "-DentityExpansionLimit=10000000",
        "--add-opens=java.base/java.lang=ALL-UNNAMED"
    )

    if not jpype.isJVMStarted():
        raise RuntimeError("LogMap JVM failed to start")



class LogMapInterface:
    """
    A python wrapper for the LogMap java interface `LogMapLLM_Interface`.
    Provides a convenient abstraction for interfacing with LogMap.
    """
    def __init__(self, src_uri: str, tgt_uri: str, task_name: str):

        self._src_uri = ontology_file_uri(src_uri)
        self._tgt_uri = ontology_file_uri(tgt_uri)
        self._task_name = task_name

        from uk.ac.ox.krr.logmap2 import LogMapLLM_Interface  # type: ignore

        self._interface = LogMapLLM_Interface(
            self._src_uri, self._tgt_uri, self._task_name,
        )


    @classmethod
    def create_and_configure(cls, src_uri: str, tgt_uri: str, task_name: str, logmap_dir: str | Path, extended_m_ask: bool, output_dir: str | Path) -> "LogMapInterface":
        """
        A convenient factory for creating and configuring a LogMapInterface
        (since we cannot overload the constructor in Python).
        """
        instance = cls(src_uri, tgt_uri, task_name)
        instance.set_parameters_dir(logmap_dir)
        instance.set_extended_questions_for_llm(extended_m_ask)
        instance.set_output_dir(output_dir)
        return instance


    @classmethod
    def create_from_cfg(cls, cfg: LogMapLLMConfig, logmap_dir: str | Path | None = None) -> "LogMapInterface":
        """
        Backwards compatability
        """
        this_logmap_dir = cfg.alignmentTask.logmap_parameters_dirpath if logmap_dir is None else logmap_dir
        instance = cls.create_and_configure(
            src_uri=cfg.alignmentTask.onto_source_filepath,
            tgt_uri=cfg.alignmentTask.onto_target_filepath,
            task_name=cfg.alignmentTask.task_name,
            logmap_dir=this_logmap_dir,
            extended_m_ask=cfg.alignmentTask.generate_extended_mappings_to_ask_oracle,
            output_dir=cfg.outputs.logmap_initial_alignment_output_dirpath,
        )
        return instance


    def set_parameters_dir(self, logmap_dir: str | Path) -> None:
        """
        Specify the LogMap directory (or a custom directory) where `parameters.txt`
        for LogMap is found. Do not include `/parameters.txt`, it should be the base dir.
        For example: `/home/user/logmap-llm/logmap/`.
        """
        logmap_dir_path = str(logmap_dir)
        if not logmap_dir_path.endswith(os.sep):
            logmap_dir_path += os.sep
        self._interface.setPathToLogMapParameters(logmap_dir_path)


    def set_output_dir(self, dirpath: str | Path) -> None:
        """
        Sets the directory LogMap writes its output (e.g., alignment, M_ask) to.
        Should be specified before step 1 (initial alignment), and step 4 (refine alignment).
        """
        self._interface.setPathForOutputMappings(str(dirpath))


    def set_extended_questions_for_llm(self, produce_extended_questions: bool) -> None:
        """Sets whether LogMap should produce the extended M_ask set."""
        self._interface.setExtendedQuestions4LLM(produce_extended_questions)


    def perform_alignment(self) -> None:
        """Perform an alignment with LogMap."""
        self._interface.performAlignment()


    def get_mappings(self):
        """Returns LogMap mappings, following an alignment."""
        return self._interface.getLogMapMappings()


    def get_mappings_for_llm(self):
        """Returns LogMap M_ask set, following an alignment."""
        return self._interface.getLogMapMappingsForLLM()


    def refine_alignment(self, oracle_predictions) -> None:
        """Refine an alignment with LogMap."""
        self._interface.performAlignmentWithLocalOracle(oracle_predictions)

    # TODO: add getters for `path_to_paramaters`, `extractExtendedQuestions4LLM`
    # and `path_to_output_mappings`; requires LogMap src changes first (these
    # instance variables are not currently public).

# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Script to generate TFLM Arduino examples ZIP file"""

import argparse
import configparser
import shutil
import subprocess
import tempfile
from enum import Enum, unique
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Union


def _run_sed_scripts(file_path: Path,
                     scripts: List[str],
                     is_dry_run: bool = True) -> None:
  """
  Run SED scripts with specified arguments against the given file.
  The file is updated in place.

  Args:
    file_path: The full path to the input file
    scripts: A list of strings, each containing a single SED script
    is_dry_run: if True, do not execute any commands

  Raises:
    CalledProcessError: command executed by the subshell had an error
  """
  if scripts == []:
    raise RuntimeError(f"No scripts specified for file {str(file_path)}")
  cmd = f"sed -e {' -e '.join(scripts)} {str(file_path)}"
  print(f"Running command: {cmd}")
  if not is_dry_run:
    try:
      result = subprocess.run(cmd,
                              shell=True,
                              check=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as ex:
      print(f"SED command failed, error: {str(ex.stderr, encoding='utf-8')}\n")
      raise ex from None
    print(f"Saving output to: {str(file_path)}")
    file_path.write_bytes(result.stdout)


def _remove_directories(paths: List[Path], is_dry_run: bool = True) -> None:
  """
  Remove directory tree(s) given list of pathnames

  Args:
    paths: A list of Path objects
    is_dry_run: if True, do not execute any commands
  """
  for dir_path in paths:
    print(f"Removing directory tree {str(dir_path)}")
    if dir_path.exists() and not is_dry_run:
      shutil.rmtree(dir_path)


def _remove_empty_directories_recursive(paths: Iterable[Path],
                                        root: Path,
                                        is_dry_run: bool = True) -> None:
  """
  Remove empty directories given list of pathnames, searching parent
  directories until reaching the root directory

  Args:
    paths: A list of Path objects
    root: The path at which to stop parent directory search
    is_dry_run: if True, do not execute any commands
  """
  empty_paths = list(filter(lambda p: list(p.glob("*")) == [], paths))
  parent_paths: Set[Path] = set()
  for dir_path in empty_paths:
    if dir_path == root:
      continue
    parent_paths.add(dir_path.parent)
    print(f"Removing empty directory {str(dir_path)}")
    if not is_dry_run:
      dir_path.rmdir()
  if len(parent_paths) > 0:
    _remove_empty_directories_recursive(parent_paths,
                                        root=root,
                                        is_dry_run=is_dry_run)


def _run_python_script(path_to_script: str,
                       args: str,
                       is_dry_run: bool = True) -> None:
  """
  Run a python script with specified arguments

  Args:
    path_to_script: The full path to the Python script
    args: a string containing all the script arguments
    is_dry_run: if True, do not execute any commands

  Raises:
    CalledProcessError: command executed by the subshell had an error
  """
  cmd = f"python3 {path_to_script} {args}"
  print(f"Running command: {cmd}")
  if not is_dry_run:
    try:
      _ = subprocess.run(cmd,
                         shell=True,
                         check=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as ex:
      print(
          f"Python script failed, error: {str(ex.stderr, encoding='utf-8')}\n")
      raise ex from None


def _create_directories(paths: List[Path], is_dry_run: bool = True) -> None:
  """
  Create directory tree(s) given list of pathnames

  Args:
    paths: A list of Path objects
    is_dry_run: if True, do not execute any commands
  """
  for dir_path in paths:
    print(f"Creating directory tree {str(dir_path)}")
    if not dir_path.is_dir() and not is_dry_run:
      dir_path.mkdir(mode=0o755, parents=True, exist_ok=True)


def _copy_files(paths: Iterable[Tuple[Path, Path]],
                is_dry_run: bool = True) -> None:
  """
  Copy files given list of source and destination Path tuples

  Args:
    paths: A list of tuples of Path objects.
    Each tuple is of the form (source, destination)
    is_dry_run: if True, do not execute any commands
  """
  #dir_path: Tuple[Path, Path]
  for from_path, to_path in paths:
    print(f"Copying {str(from_path)} to {str(to_path)}")
    if not is_dry_run:
      shutil.copy2(from_path, to_path)


class ArduinoProjectGenerator:
  """
  Generate the TFLM Arduino library ZIP file
  """

  #
  # private enums
  #

  @unique
  class Manifest(Enum):
    ADD = "Add",
    REMOVE = "Remove",
    SPECIAL_REPO = "Special Repo",
    SPECIAL_BASE = "Special Base"
    PATCH_SED = "Patch Sed"

  #
  # private methods
  #

  def __init__(self) -> None:
    args = self._parse_arguments().parse_args()
    self._base_dir = Path(args.base_dir)
    if args.output_dir is None:
      self._output_dir = Path(tempfile.gettempdir()) / "tflm_arduino"
    else:
      self._output_dir = Path(args.output_dir)
    self._is_dry_run: bool = args.is_dry_run
    if args.manifest_file is None:
      self._manifest_path = Path("scripts/MANIFEST.ini")
    else:
      self._manifest_path = Path(args.manifest_file)

    # generate list of examples by inspecting repo examples directory
    self._examples: List[str] = []
    for path in Path("examples").glob("*"):
      if path.is_dir:
        self._examples.append(path.name)

    # parse manifest file
    manifest = self._parse_manifest()
    self._add_list: List[Path] = manifest[self.Manifest.ADD]
    self._remove_list: List[Path] = manifest[self.Manifest.REMOVE]
    self._special_repo_list: List[Tuple[Path, Path]] = manifest[
        self.Manifest.SPECIAL_REPO]
    self._special_base_list: List[Tuple[Path, Path]] = manifest[
        self.Manifest.SPECIAL_BASE]
    self._patch_sed_list: List[Tuple[List[Path], List[str]]] = manifest[
        self.Manifest.PATCH_SED]

  def _parse_manifest(self) -> Dict[Manifest, List]:
    if not self._manifest_path.exists():
      raise RuntimeError(
          f"Unable to locate manifest file {str(self._manifest_path)}")
    manifest: Dict[self.Manifest, List] = dict()
    parser = configparser.ConfigParser()
    _ = parser.read(self._manifest_path)
    files = filter(None, parser["Add Files"]["files"].splitlines())
    manifest[self.Manifest.ADD] = list(map(Path, files))
    files = filter(None, parser["Remove Files"]["files"].splitlines())
    manifest[self.Manifest.REMOVE] = list(map(Path, files))
    manifest[self.Manifest.SPECIAL_REPO] = []
    manifest[self.Manifest.SPECIAL_BASE] = []
    manifest[self.Manifest.PATCH_SED] = []
    for section in parser.sections():
      if section in ["Add Files", "Remove Files", "DEFAULT"]:
        continue
      # check if patch with sed scripts
      sed_scripts = parser[section].get("sed_scripts")
      if sed_scripts is not None:
        sed_scripts = filter(None, sed_scripts.splitlines())
        sed_scripts = list(sed_scripts)
        files = filter(None, parser[section]["files"].splitlines())
        files = map(lambda file: Path(file.strip()), files)
        files = list(files)
        manifest[self.Manifest.PATCH_SED].append((files, sed_scripts))
      else:
        to_file = parser[section]["to"]
        from_file = parser[section].get("from_repo")
        if from_file is None:
          from_file = parser[section]["from"]
          key = self.Manifest.SPECIAL_BASE
        else:
          key = self.Manifest.SPECIAL_REPO
        path_tuple = (Path(from_file.strip()), Path(to_file.strip()))
        manifest[key].append(path_tuple)

    return manifest

  def _parse_arguments(self) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Script for TFLM Arduino project generation")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Output directory for generated TFLM tree")
    parser.add_argument("--base_dir",
                        type=str,
                        required=True,
                        help="Base directory for generating TFLM tree")
    parser.add_argument("--manifest_file",
                        type=str,
                        default=None,
                        help="Manifest file path (relative or absolute)")
    parser.add_argument("--is_dry_run",
                        default=False,
                        action="store_true",
                        help="Show commands only (no execution)")
    return parser

  def _clean_output_directory(self) -> None:
    dirs_to_remove = [self._output_dir]
    _remove_directories(dirs_to_remove, is_dry_run=self._is_dry_run)
    zip_path = self._output_dir.with_suffix(".zip")
    print(f"Removing ZIP file: {str(zip_path)}")
    if zip_path.exists() and not self._is_dry_run:
      zip_path.unlink()

  def _create_output_directories(
      self, all_path_pairs: List[Tuple[Path, Path]]) -> None:
    # generate full list of source tree directories
    # collect relative destination paths and sort relative paths
    set_relative_subdirs: Set[Path] = {
        path[1].parent for path in all_path_pairs if path[1].parent != Path(".")
    }
    relative_subdirs = list(set_relative_subdirs)
    relative_subdirs.sort()

    # filter out common parents
    def _filter_func(pair: Tuple[int, Path]):
      index = pair[0]
      if index == len(relative_subdirs) - 1:
        return True
      elif pair[1] not in relative_subdirs[index + 1].parents:
        return True
      else:
        return False

    filtered_subdirs: List[Tuple[int, Path]] = list(
        filter(_filter_func, enumerate(relative_subdirs)))
    # convert from enumerated tuples back into list of Path objects
    if filtered_subdirs != []:
      relative_subdirs = list(list(zip(*filtered_subdirs))[1])
    else:
      relative_subdirs = []

    # convert relative paths to full destination paths
    dst_subdirs = [self._output_dir / path for path in relative_subdirs]
    _create_directories(dst_subdirs, is_dry_run=self._is_dry_run)

  def _copy_no_transform(
      self,
      relative_paths: List[Tuple[Path, Path]],
      relative_to: Path = Path(".")
  ) -> None:

    full_paths = [(relative_to / item[0], self._output_dir / item[1])
                  for item in relative_paths]
    _copy_files(full_paths, is_dry_run=self._is_dry_run)

  def _copy_with_transform(
      self,
      path_pairs: List[Tuple[Path, Path]],
      headers_dict: Dict[Union[str, None], str],
      relative_to: Path = Path(".")
  ) -> None:

    script_path = "sync/transform_source.py"

    # transform all source and header files
    for relative_paths in path_pairs:
      dst_path = self._output_dir / relative_paths[1]
      src_path = relative_to / relative_paths[0]
      if relative_paths[1].parts[0] == "examples":
        third_party_headers = headers_dict[relative_paths[1].parts[1]]
        if relative_paths[1].suffix == ".ino":
          is_example_ino = True
          is_example_source = False
        else:
          is_example_ino = False
          is_example_source = True
      else:
        third_party_headers = headers_dict[None]
        is_example_source = False
        is_example_ino = False

      args = "--platform=arduino"
      if is_example_ino:
        args += " --is_example_ino"
      elif is_example_source:
        args += " --is_example_source"
      args += f' --third_party_headers="{third_party_headers}"'
      args += f" < {str(src_path)} > {str(dst_path)}"
      _run_python_script(script_path, args=args, is_dry_run=self._is_dry_run)

  def _glob_expand(self, relative_dir: Path, files: List[Path]) -> List[Path]:
    result_list: List[Path] = []
    for file in files:
      expanded = list(relative_dir.glob(str(file)))
      if not expanded:
        print(f"*** Unknown path: {str(relative_dir / file)}")
        continue
      elif len(expanded) == 1:
        if expanded[0].is_dir():
          expanded = expanded[0].glob("**/*")

      expanded = filter(lambda p: not p.is_dir(), expanded)
      # future: use transform_suffixes=
      result_list.extend(expanded)

    return result_list

  def _patch_with_sed(self, patches: List[Tuple[List[Path],
                                                List[str]]]) -> None:
    for files, scripts in patches:
      glob_expanded_files = self._glob_expand(self._output_dir, files)
      print(glob_expanded_files)
      for file in glob_expanded_files:
        _run_sed_scripts(file, scripts=scripts, is_dry_run=self._is_dry_run)

  def _fix_subdirectories(self) -> None:
    script_path = "sync/fix_arduino_subfolders.py"
    args = str(self._output_dir)
    _run_python_script(script_path, args, is_dry_run=self._is_dry_run)

  def _generate_header_list(
      self, example: Union[str, None],
      all_path_pairs: List[Tuple[Path, Path]]) -> List[str]:
    result_list: List[str] = []
    for path in all_path_pairs:
      if path[1].suffix != ".h":
        continue
      if example is not None:
        # need the headers for this example
        relative_path = Path("examples") / example
        if relative_path in path[1].parents:
          s = str(path[1])
          result_list.append(s)
      # add third-party headers
      relative_path = Path("src/third_party")
      if relative_path in path[1].parents:
        s = str(relative_path.parts[1] / path[1].relative_to(relative_path))
        result_list.append(s)

    return result_list

  def _generate_headers_dict(
      self, all_path_pairs: List[Tuple[Path,
                                       Path]]) -> Dict[Union[str, None], str]:
    headers_dict: Dict[Union[str, None], str] = {
        example: " ".join(
            self._generate_header_list(example, all_path_pairs=all_path_pairs))
        for example in self._examples
    }
    headers_dict[None] = " ".join(
        self._generate_header_list(None, all_path_pairs=all_path_pairs))
    return headers_dict

  def _generate_base_paths_relative(
      self, generate_transform_paths: bool) -> List[Tuple[Path, Path]]:

    # generate set of all files from base directory
    all_files = self._base_dir.glob("**/*")
    # filter out directories
    all_files = filter(lambda p: not p.is_dir(), all_files)
    special_path_dict = dict(self._special_base_list)

    # generate relative source/destination pairs
    relative_path_pairs: List[Tuple[Path, Path]] = []
    for file in all_files:
      src_path = Path(file).relative_to(self._base_dir)
      # check if in manifest special base or remove list
      if special_path_dict.get(src_path) is not None:
        continue
      elif src_path in self._remove_list:
        # file in remove list, skip {src_path}
        continue
      elif any([parent in self._remove_list for parent in src_path.parents]):
        # parent directory in remove list, skip {src_path}
        continue
      # tensorflow/ and third_party/ will be subdirectories of src/
      if src_path.parts[0] in ["third_party", "tensorflow"]:
        dst_path = "src" / src_path
      else:
        dst_path = src_path

      # check for .cc to .cpp rename
      # future: apply suffixes_mapping= instead
      if dst_path.suffix == ".cc":
        dst_path = dst_path.with_suffix(".cpp")

      # add new tuple(src,dst) to list
      relative_path_pairs.append((src_path, dst_path))

    # add all manifest special base paths
    for paths in self._special_base_list:
      from_path = paths[0]
      to_path = paths[1]
      # future: apply suffixes_mapping=
      relative_path_pairs.append((from_path, to_path))

    # filter for (non)transformable
    # future: use transform_suffixes=
    xform_suffixes = [".c", ".cc", ".cpp", ".h", ".ino"]
    relative_path_pairs = list(
        filter(
            lambda pair: generate_transform_paths ==
            (pair[0].suffix in xform_suffixes), relative_path_pairs))

    return relative_path_pairs

  def _generate_repo_paths_relative(
      self, generate_transform_paths: bool) -> List[Tuple[Path, Path]]:

    relative_path_pairs: List[Tuple[Path, Path]] = []
    for path in self._add_list:
      # future: apply suffixes_mapping=
      if path.is_dir():
        path_glob_list = list(
            filter(lambda p: not p.is_dir(), path.glob("**/*")))
        relative_path_pairs.extend(zip(path_glob_list, path_glob_list))
      else:
        relative_path_pairs.append((path, path))
    for paths in self._special_repo_list:
      from_path = paths[0]
      to_path = paths[1]
      # future: apply suffixes_mapping=
      relative_path_pairs.append((from_path, to_path))

    # filter for (non)transformable
    # future: use transform_suffixes=
    xform_suffixes = [".c", ".cc", ".cpp", ".h", ".ino"]
    third_party_path = Path("src/third_party")

    def filter_func(path_pair: Tuple[Path, Path]):
      is_third_party = str(third_party_path) in str(path_pair[0])
      can_xform = path_pair[0].suffix in xform_suffixes
      if generate_transform_paths:
        # doing transforms, only process third-party xforms
        return is_third_party and can_xform
      else:
        # not doing transforms, skip third-party xforms
        return not (is_third_party and can_xform)

    relative_path_pairs = list(filter(filter_func, relative_path_pairs))

    return relative_path_pairs

  def _remove_empty_directories(self) -> None:
    paths = self._output_dir.glob("**")
    _remove_empty_directories_recursive(paths,
                                        root=self._output_dir,
                                        is_dry_run=self._is_dry_run)

  #
  # public methods
  #

  def generate_tree(self) -> None:
    """
    Execute all steps to create TFLM Arduino ZIP file
    """
    self._clean_output_directory()
    base_xform_paths = self._generate_base_paths_relative(True)
    base_copy_paths = self._generate_base_paths_relative(False)
    repo_xform_paths = self._generate_repo_paths_relative(True)
    repo_copy_paths = self._generate_repo_paths_relative(False)
    all_path_pairs = [
        *base_xform_paths,
        *base_copy_paths,
        *repo_xform_paths,
        *repo_copy_paths,
    ]
    self._create_output_directories(all_path_pairs)
    headers_dict = self._generate_headers_dict(all_path_pairs)
    self._copy_with_transform(base_xform_paths,
                              headers_dict=headers_dict,
                              relative_to=self._base_dir)
    self._copy_with_transform(repo_xform_paths, headers_dict=headers_dict)
    self._copy_no_transform(base_copy_paths, relative_to=self._base_dir)
    self._copy_no_transform(repo_copy_paths)
    self._fix_subdirectories()
    self._patch_with_sed(self._patch_sed_list)
    self._remove_empty_directories()


if __name__ == "__main__":
  generator = ArduinoProjectGenerator()
  generator.generate_tree()
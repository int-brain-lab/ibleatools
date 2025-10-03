
import luigi
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import os
import logging
import numpy as np

# Meta variables for the workflow
PROJECT_NAME = "psychedlics"  # Change this to your project name
OUTPUT_DIR = Path(f"/mnt/sdceph/users/prai1/data/projects/{PROJECT_NAME}")
VENV_PATH = "/mnt/home/prai1/projects/passive_ephys/.venv/bin/activate"
TEMPLATE_COMPUTATION_PATH = Path(__file__).parent / "computation_template.py"
TEMPLATE_RUNPROGRAM_PATH = Path(__file__).parent / "Runprogram_template.sh"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GetPidList(luigi.Task):
    """Task to get list of pids, eids and probe_names and save as CSV."""
    
    def output(self):
        return luigi.LocalTarget(str(OUTPUT_DIR / "pids_eids_probes.csv"))
    
    def run(self):
        """Get the list of pids, eids and probe_names."""
        logger.info("Getting PID list...")
        
        # Call your implemented function
        df = get_pid_list()
        
        # Validate output data types
        required_columns = ['pid', 'eid', 'probe_name']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Validate data types
        #TODO
        
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(self.output().path, index=False)
        logger.info(f"PID list saved to {self.output().path}")


class CreateSnippetsFile(luigi.Task):
    """Task to create snippets dataframe with t_starts and duration."""
    
    def requires(self):
        return GetPidList()
    
    def output(self):
        return luigi.LocalTarget(str(OUTPUT_DIR / f"{PROJECT_NAME}_snippets_df.csv"))
    
    def run(self):
        """Create snippets dataframe from PID list."""
        logger.info("Creating snippets file...")
        
        # Read the input CSV
        df = pd.read_csv(self.input().path)
        
        # Call your implemented function
        snippets_df = create_snippets_file(df)
        
        # Validate output data types
        required_columns = ['pid', 'eid', 'probe_name', 't_start', 'duration']
        if not all(col in snippets_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        # Validate data types
        #TODO
        
        # Save to CSV
        snippets_df.to_csv(self.output().path, index=False)
        logger.info(f"Snippets file saved to {self.output().path}")


class CreateComputationFile(luigi.Task):
    """Task to create computation.py file from template."""
    
    def output(self):
        return luigi.LocalTarget(str(OUTPUT_DIR / "computation.py"))
    
    def run(self):
        """Create computation.py file from template."""
        logger.info("Creating computation file...")
        
        # Call your implemented function
        computation_content = create_computation_file()
        
        # Validate output
        if not isinstance(computation_content, str):
            raise ValueError("Computation file content should be a string")
        
        # Ensure OUTPUT_DIR exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        with open(self.output().path, 'w') as f:
            f.write(computation_content)
        
        logger.info(f"Computation file created at {self.output().path}")


class CreateRunProgramFile(luigi.Task):
    """Task to create Runprogram.sh file."""
    
    def requires(self):
        return CreateComputationFile()
    
    def output(self):
        return luigi.LocalTarget(str(OUTPUT_DIR / "Runprogram.sh"))
    
    def run(self):
        """Create Runprogram.sh file from template."""
        logger.info("Creating Runprogram.sh file...")
        
        # Read the template file
        template_path = TEMPLATE_RUNPROGRAM_PATH
        if not template_path.exists():
            raise FileNotFoundError(f"Runprogram template file not found at {template_path}")
        
        with open(template_path, 'r') as f:
            run_program_template = f.read()
        
        # Replace placeholders in the template
        run_program_content = run_program_template.replace(
            '{VENV_PATH}', VENV_PATH
        ).replace(
            '{OUTPUT_DIR}', str(OUTPUT_DIR)
        )
        
        # Ensure OUTPUT_DIR exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        with open(self.output().path, 'w') as f:
            f.write(run_program_content)
        
        # Make it executable
        os.chmod(self.output().path, 0o755)
        
        logger.info(f"Runprogram.sh created at {self.output().path}")


class CreateTaskFile(luigi.Task):
    """Task to create the final task file."""
    
    def requires(self):
        return {
            'snippets': CreateSnippetsFile(),
            'run_program': CreateRunProgramFile()
        }
    
    def output(self):
        return luigi.LocalTarget(str(OUTPUT_DIR / "Full_Task_file"))
    
    def run(self):
        """Create the task file."""
        logger.info("Creating task file...")
        
        # Read snippets file
        snippets_df = pd.read_csv(self.input()['snippets'].path)
        
        # Get run program path
        run_program_path = self.input()['run_program'].path

        run_program_path = Path(run_program_path).as_posix()

        outp_file = OUTPUT_DIR / 'Full_Task_file'
        
        # Call your implemented function
        task_file_content = create_task_file(snippets_df, outp_file, run_program_path)
        
        # Validate output
        if not Path(outp_file).exists():
            raise ValueError("Task file was not created")
        
        logger.info(f"Task file created at {self.output().path}")


class WorkflowPipeline(luigi.Task):
    """Main workflow pipeline that runs all tasks."""
    
    def requires(self):
        return CreateTaskFile()
    
    def output(self):
        return luigi.LocalTarget(str(OUTPUT_DIR / "workflow_complete.txt"))
    
    def run(self):
        """Run the complete workflow."""
        logger.info("Workflow pipeline completed successfully!")
        
        # Create a completion marker
        with open(self.output().path, 'w') as f:
            f.write("Workflow completed successfully\n")
            f.write(f"Project: {PROJECT_NAME}\n")
            f.write(f"Output directory: {OUTPUT_DIR}\n")
            f.write(f"Generated files:\n")
            f.write(f"  - pids_eids_probes.csv\n")
            f.write(f"  - {PROJECT_NAME}_snippets_df.csv\n")
            f.write(f"  - computation.py\n")
            f.write(f"  - Runprogram.sh\n")
            f.write(f"  - Full_Task_file\n")


# Placeholder functions - implement these with your actual logic
def get_pid_list() -> pd.DataFrame:
    """
    Get the list of pids, eids and probe_names.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['pid', 'eid', 'probe_name']
    """

    from one.api import ONE

    one = ONE()
    # Rest query for getting psychedelics insertions
    insertions = one.alyx.rest('insertions', 'list', django='session__projects__name__icontains,psychedelics')

    # Get pids from insertions
    pids, alyx_pids = [item["id"] for item in insertions], insertions

    # Get the corresponding eids, and probe_names
    df= pd.DataFrame([{'pid':val['id'],'eid':val['session'],'probe_name':val['name']} for val in alyx_pids],columns=['pid','eid','probe_name'])

    #TODO - Add exclude pids
    return df


def create_snippets_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create snippets dataframe from PID list with t_starts and duration.
    """
    DURATION = 5
    list_df = []
    for index, row in df.iterrows():
        df_temp = pd.DataFrame(columns=['pid', 'eid', 'probe_name', 'snippet_index', 't_start', 'duration'])
        # Set the start times for the snippets.
        start_times = np.array([300,600,900,1200,1500])
        df_temp['snippet_index'] = np.arange(len(start_times))
        df_temp['t_start'] = start_times
        df_temp['pid'] = row['pid']
        df_temp['eid'] = row['eid']
        df_temp['probe_name'] = row['probe_name']
        df_temp['duration'] = DURATION
        list_df.append(df_temp)

    df = pd.concat(list_df)
    return df
    # df.to_csv(OUTPUT_DIR / 'psychedlics_snippets_df.csv', index=False)


def create_computation_file() -> str:
    """
    Create computation.py file content from template.
    
    Returns:
        str: Content of the computation.py file
    """
    # Read the template file
    template_path = TEMPLATE_COMPUTATION_PATH
    if template_path.exists():
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Replace the OUTPUT_DIR in the template with the actual OUTPUT_DIR
        computation_content = template_content.replace(
            'OUTPUT_DIR = Path("/mnt/sdceph/users/prai1/data/projects/psychedlics/output/")',
            f'OUTPUT_DIR = Path("{OUTPUT_DIR}/output/")'
        )
        return computation_content
    else:
        raise FileNotFoundError(f"Template file not found at {template_path}")

def create_task_file(inp_file, outp_file, run_program_path="Runprogram.sh"):
    # Read the CSV file

    if isinstance(inp_file, pd.DataFrame):
        df = inp_file
    elif isinstance(inp_file, (str,Path)):
        df = pd.read_csv(inp_file)
    else:
        raise ValueError(f"Invalid input type: {type(inp_file)}")
    
    # Create the task file
    task_file_path = outp_file
    
    # Generate command lines for each row
    with open(task_file_path, 'w') as f:
        for _, row in df.iterrows():
            command = f"source {run_program_path} --pid {row['pid']} --eid {row['eid']} --probe_name {row['probe_name']} --start_time {row['t_start']} --duration {row['duration']}\n"
            f.write(command)
    
    print(f"Task file created at: {task_file_path}")


# Function to run the workflow
def run_workflow():
    """
    Run the complete workflow pipeline.
    """
    
    luigi.build([WorkflowPipeline()], local_scheduler=True)

if __name__ == "__main__":
    run_workflow()


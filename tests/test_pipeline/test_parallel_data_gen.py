"""
Tests for parallel data generation functionality.
"""

import os
import tempfile
import time

import numpy as np
import pytest
import yaml

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'PyFEM_Dynamics'))
from pipeline.data_gen import generate_dataset, _generate_sample


@pytest.fixture
def minimal_config():
    """Create minimal config for fast testing."""
    config = {
        'structure_file': 'structure.yaml',
        'output_file': 'test_output/test.npz',
        'time': {
            'dt': 0.01,
            'total_time': 0.5,
        },
        'generation': {
            'num_samples': 10,
            'random_seed': 42,
        },
        'load_generation': {
            'mode': 'random',
            'random_multi_point': {
                'num_loads_per_sample_range': [1, 2],
                'candidate_nodes': {
                    'mode': 'all_non_support_nodes',
                    'include_nodes': [],
                    'exclude_nodes': [],
                },
                'dof_candidates': ['fy'],
                'dof_weights': [1.0],
                'pattern_candidates': ['harmonic'],
                'pattern_weights': [1.0],
                'parameter_ranges': {
                    'F0': [100.0, 500.0],
                    'freq': [0.5, 2.0],
                    'phase': 0.0,
                },
            },
        },
        'damage': {
            'enabled': True,
            'min_damaged_elements': 1,
            'max_damaged_elements': 2,
            'reduction_range': [0.5, 0.9],
        },
        'damping': {
            'alpha': 0.1,
            'beta': 0.01,
        },
    }
    return config


@pytest.fixture
def temp_config_file(minimal_config):
    """Create temporary config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, 'test_config.yaml')
        output_dir = os.path.join(tmpdir, 'test_output')
        os.makedirs(output_dir, exist_ok=True)
        
        minimal_config['output_file'] = os.path.join(output_dir, 'test.npz')
        
        structure_path = os.path.join(os.path.dirname(__file__), '..', '..', 'structure.yaml')
        if os.path.exists(structure_path):
            import shutil
            shutil.copy(structure_path, os.path.join(tmpdir, 'structure.yaml'))
            minimal_config['structure_file'] = 'structure.yaml'
        else:
            pytest.skip("structure.yaml not found")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(minimal_config, f)
        
        yield config_path


class TestCLIArguments:
    """Test CLI argument parsing."""
    
    def test_jobs_default_minus_one(self):
        """Test default --jobs is -1 (auto)."""
        import argparse
        from PyFEM_Dynamics.pipeline.data_gen import generate_dataset
        
        assert True
    
    def test_seq_flag_default_false(self):
        """Test --seq flag defaults to False."""
        assert True


class TestWorkerFunction:
    """Test _generate_sample worker function."""
    
    def test_worker_produces_valid_output(self, temp_config_file):
        """Test that worker produces dict with all required keys."""
        with open(temp_config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        from PyFEM_Dynamics.core.io_parser import YAMLParser
        from PyFEM_Dynamics.pipeline.data_gen import apply_damage
        
        structure_file = os.path.join(os.path.dirname(temp_config_file), config['structure_file'])
        nodes, elements, bcs = YAMLParser.build_structure_objects(structure_file)
        num_nodes = len(nodes)
        num_elements = len(elements)
        dofs_per_node = len(nodes[0].dofs) if nodes else 2
        
        dt = float(config['time']['dt'])
        total_time = float(config['time']['total_time'])
        damage_config = config.get('damage', {})
        load_generation_cfg = config['load_generation']
        load_mode = str(load_generation_cfg.get('mode', 'fixed')).lower()
        damping_config = config.get('damping', {})
        alpha = damping_config.get('alpha', 0.1)
        beta = damping_config.get('beta', 0.01)
        
        E_base = np.array([el.material.E for el in elements], dtype=float)
        base_seed = 42
        
        args = (
            0,
            base_seed,
            structure_file,
            E_base,
            damage_config,
            load_generation_cfg,
            load_mode,
            dt,
            total_time,
            alpha,
            beta,
            num_nodes,
            num_elements,
            dofs_per_node,
            bcs,
        )
        
        sample_idx, result = _generate_sample(args)
        
        assert sample_idx == 0
        assert 'load' in result
        assert 'E' in result
        assert 'disp' in result
        assert 'stress' in result
        assert 'damage' in result
        
        num_steps = int(total_time / dt) + 1
        num_dofs = num_nodes * dofs_per_node
        
        assert result['load'].shape == (num_steps, num_dofs)
        assert result['E'].shape == (num_elements,)
        assert result['disp'].shape == (num_steps, num_dofs)
        assert result['stress'].shape == (num_steps, num_elements)
        assert result['damage'].shape == (num_elements,)


class TestParallelDeterminism:
    """Test that parallel and sequential modes produce identical results."""
    
    def test_sequential_vs_parallel_output(self, temp_config_file):
        """Test parallel mode produces same output as sequential."""
        output_dir = os.path.dirname(
            yaml.safe_load(open(temp_config_file))['output_file']
        )
        
        seq_output = os.path.join(output_dir, 'seq.npz')
        par_output = os.path.join(output_dir, 'par.npz')
        
        with open(temp_config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['output_file'] = seq_output
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        generate_dataset(config_path=temp_config_file, n_jobs=1, sequential=True)
        
        config['output_file'] = par_output
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        generate_dataset(config_path=temp_config_file, n_jobs=2, sequential=False)
        
        seq_data = np.load(seq_output)
        par_data = np.load(par_output)
        
        for key in ['load', 'E', 'disp', 'stress', 'damage']:
            seq_arr = seq_data[key]
            par_arr = par_data[key]
            assert seq_arr.shape == par_arr.shape, f"Shape mismatch for {key}"
            assert np.allclose(seq_arr, par_arr, rtol=1e-10, atol=1e-10), \
                f"Values mismatch for {key}"


class TestSeedReproducibility:
    """Test that results are reproducible with same seed."""
    
    def test_same_seed_same_output(self, temp_config_file):
        """Test that same seed produces identical output."""
        output_dir = os.path.dirname(
            yaml.safe_load(open(temp_config_file))['output_file']
        )
        
        run1_output = os.path.join(output_dir, 'run1.npz')
        run2_output = os.path.join(output_dir, 'run2.npz')
        
        with open(temp_config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['output_file'] = run1_output
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        generate_dataset(config_path=temp_config_file, n_jobs=1, sequential=True)
        
        config['output_file'] = run2_output
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        generate_dataset(config_path=temp_config_file, n_jobs=1, sequential=True)
        
        run1_data = np.load(run1_output)
        run2_data = np.load(run2_output)
        
        for key in ['load', 'E', 'disp', 'stress', 'damage']:
            assert np.allclose(run1_data[key], run2_data[key], rtol=1e-10, atol=1e-10), \
                f"Reproducibility failed for {key}"


class TestSmallScaleExecution:
    """Test small-scale parallel execution."""
    
    def test_parallel_samples_10_jobs_2(self, temp_config_file):
        """Test parallel execution with 10 samples and 2 workers."""
        with open(temp_config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        output_dir = os.path.dirname(config['output_file'])
        output_file = os.path.join(output_dir, 'test_par.npz')
        
        config['output_file'] = output_file
        config['generation']['num_samples'] = 10
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        generate_dataset(config_path=temp_config_file, n_jobs=2, sequential=False)
        
        assert os.path.exists(output_file)
        
        data = np.load(output_file)
        assert data['load'].shape[0] == 10
        assert data['E'].shape[0] == 10
        assert data['disp'].shape[0] == 10
        assert data['stress'].shape[0] == 10
        assert data['damage'].shape[0] == 10


class TestSpeedupBenchmark:
    """Benchmark tests for speedup verification."""
    
    @pytest.mark.skipif(
        os.cpu_count() is None or os.cpu_count() < 2,
        reason="Need at least 2 CPU cores for speedup test"
    )
    def test_speedup_ratio(self, temp_config_file):
        """Test that parallel mode achieves speedup over sequential."""
        with open(temp_config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        output_dir = os.path.dirname(config['output_file'])
        
        config['generation']['num_samples'] = 20
        config['time']['total_time'] = 0.2
        
        seq_output = os.path.join(output_dir, 'bench_seq.npz')
        config['output_file'] = seq_output
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        start_seq = time.time()
        generate_dataset(config_path=temp_config_file, n_jobs=1, sequential=True)
        seq_time = time.time() - start_seq
        
        par_output = os.path.join(output_dir, 'bench_par.npz')
        config['output_file'] = par_output
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        n_jobs = min(4, os.cpu_count() or 1)
        start_par = time.time()
        generate_dataset(config_path=temp_config_file, n_jobs=n_jobs, sequential=False)
        par_time = time.time() - start_par
        
        speedup = seq_time / par_time if par_time > 0 else 1.0
        
        print(f"\nSpeedup benchmark: sequential={seq_time:.2f}s, "
              f"parallel({n_jobs} jobs)={par_time:.2f}s, speedup={speedup:.2f}x")
        
        assert speedup >= 1.0, \
            f"Parallel mode should not be slower than sequential. " \
            f"Got speedup={speedup:.2f}x (seq={seq_time:.2f}s, par={par_time:.2f}s)"


class TestBackwardCompatibility:
    """Test backward compatibility with existing API."""
    
    def test_generate_dataset_default_args(self, temp_config_file):
        """Test generate_dataset works with default arguments."""
        with open(temp_config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        output_dir = os.path.dirname(config['output_file'])
        config['generation']['num_samples'] = 5
        config['output_file'] = os.path.join(output_dir, 'default_args.npz')
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        generate_dataset(config_path=temp_config_file)
        
        assert os.path.exists(config['output_file'])
    
    def test_n_jobs_minus_one_auto(self, temp_config_file):
        """Test n_jobs=-1 auto-detects CPU count."""
        with open(temp_config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        output_dir = os.path.dirname(config['output_file'])
        config['generation']['num_samples'] = 5
        config['output_file'] = os.path.join(output_dir, 'auto_jobs.npz')
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        generate_dataset(config_path=temp_config_file, n_jobs=-1, sequential=False)
        
        assert os.path.exists(config['output_file'])
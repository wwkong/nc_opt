% Unit tests for the Oracle class.

%% Oracle constructors.
classdef oracle_unit_tests < matlab.unittest.TestCase
  % Tests
  methods (Test) 
    function constructor_default(testCase)
      % Default constructor.
      oracle0 = Oracle();
      testCase.assertClass(oracle0, 'Oracle');
    end
    function constructor_4arg(testCase)
      % 4-argument construction.
      oracle1 = Oracle(@(x) norm(x)^2/2, @(x) 0, @(x) x, @(x, lam) x);
      testCase.assertClass(oracle1, 'Oracle');
    end    
  end
end
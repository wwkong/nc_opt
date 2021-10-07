run('../../init.m');
testCase = oracle_unit_tests;
results = testCase.run;
disp(results);

foo = Oracle(@(x) norm(x)^2/2, @(x) 0, @(x) x, @(x, lam) x);
foo.scale(pi);
foo.eval(2);
disp(ismembertol(foo.f_s(), pi * 2));

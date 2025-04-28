from lark import Lark, Transformer
from z3 import *
from typing import Dict, List, Any, Tuple, Optional

# === FOL Parser and Solver ===

# Grammar for First-Order Logic (FOL)
fol_grammar = r"""
    ?start: formula

    ?formula: implication

    ?implication: iff
                | iff "->" implication   -> implies
                | iff "→" implication    -> implies

    ?iff: xor
        | xor "<->" iff         -> iff
        | xor "↔" iff           -> iff

    ?xor: disjunction
        | disjunction "⊕" xor   -> xor

    ?disjunction: conjunction
                | conjunction "or" disjunction -> or_op
                | conjunction "∨" disjunction  -> or_op

    ?conjunction: negation
                | negation "and" conjunction  -> and_op
                | negation "∧" conjunction    -> and_op

    ?negation: "not" negation           -> not_op
            | "¬" negation             -> not_op
            | quantifier
            | atom

    ?quantifier: "ForAll" VARIABLE "(" formula ")"     -> forall
            | "Exists" VARIABLE "(" formula ")"     -> exists
            | "∀" VARIABLE "(" formula ")"          -> forall
            | "∃" VARIABLE "(" formula ")"          -> exists
            | "ForAll" "(" VARIABLE "," formula ")" -> forall
            | "Exists" "(" VARIABLE "," formula ")" -> exists

    ?atom: comparison
        | predicate
        | "(" formula ")"

    comparison: VARIABLE "=" NUMBER      -> equals_num
            | VARIABLE "==" NUMBER     -> equals_num
            | VARIABLE "<" NUMBER      -> less_than
            | VARIABLE ">" NUMBER      -> greater_than
            | VARIABLE "<=" NUMBER     -> less_equal
            | VARIABLE ">=" NUMBER     -> greater_equal
            | VARIABLE "!=" NUMBER     -> not_equal
            | VARIABLE "≠" NUMBER      -> not_equal
            | VARIABLE "≤" NUMBER      -> less_equal
            | VARIABLE "≥" NUMBER      -> greater_equal
            | VARIABLE "=" STRING      -> equals_str
            | VARIABLE "==" STRING     -> equals_str
            | VARIABLE "!=" STRING     -> not_equal_str
            | VARIABLE "≠" STRING      -> not_equal_str

    predicate: PREDICATE "(" [term ("," term)*] ")"    -> predicate_with_args
            | PREDICATE                               -> predicate_without_args

    ?term: VARIABLE        -> term_variable
        | CONSTANT        -> term_constant

    PREDICATE: /[A-Z][a-zA-Z0-9_]*/
    VARIABLE: /[a-z][a-z0-9_%-]*/         // Lowercase first char
    CONSTANT: /[A-Za-z0-9._%-]+/
    NUMBER: /[0-9]+(\.[0-9]+)?/          // Integer or decimal number
    STRING: /("[^"]*")|('[^']*')/        // String literals enclosed in quotes

    %import common.WS
    %ignore WS
"""


# === Parser Wrapper ===
class FOLLarkParser:
    def __init__(self):
        self.lark_parser = Lark(fol_grammar, start="start", parser="lalr")
        self.transformer = FOLTransformer()
        self.solver = Solver()
        self.entailment_checker = None
        self._init_entailment_checker()

    def _init_entailment_checker(self):
        self.entailment_checker = EnhancedFOLChecker(self)

    def parse(self, fol_string: str):
        try:
            tree = self.lark_parser.parse(fol_string)
        except:
            fol_string = "JQKA(x)"
            tree = self.lark_parser.parse(fol_string)
        return self.transformer.transform(tree)

    def parse_fol_string(self, fol_string: str) -> z3.BoolRef:
        return self.parse(fol_string)

    def add_assertion(self, fol_string: str) -> None:
        formula = self.parse(fol_string)
        self.solver.add(formula)

    def check_sat(self) -> z3.CheckSatResult:
        return self.solver.check()

    def get_model(self) -> Optional[z3.ModelRef]:
        if self.check_sat() == sat:
            return self.solver.model()
        return None

    def check_entailment(self, conclusion: str) -> Tuple[bool, Optional[z3.ModelRef], str]:
        return self.entailment_checker.check_entailment(conclusion)

    def check_logical_equivalence(self, formula1: str, formula2: str) -> Tuple[bool, str]:
        return self.entailment_checker.check_logical_equivalence(formula1, formula2)

    def check_validity(self, formula: str) -> Tuple[bool, Optional[z3.ModelRef], str]:
        return self.entailment_checker.check_validity(formula)

    def generate_all_models(self, max_models: int = 10) -> List[Tuple[z3.ModelRef, str]]:
        return self.entailment_checker.generate_all_models(max_models)

    def explain_counterexample(self, model: z3.ModelRef, formula) -> str:
        return self.entailment_checker._generate_counterexample(model, formula)

    def get_unsat_core(self, conclusion: str) -> Tuple[bool, List[int], str]:
        return self.entailment_checker.get_unsat_core(conclusion)

# === FOL to Z3 Transformer ===


class FOLTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.variables: Dict[str, ExprRef] = {}
        self.predicates: Dict[str, FuncDeclRef] = {}
        self.comparison_variables: Dict[str, bool] = {}

    def get_var(self, name):
        if name not in self.variables:
            self.variables[name] = Const(name, BoolSort())
        return self.variables[name]

    def get_predicate(self, name, arity=0):
        key = f"{name}_{arity}"
        if key not in self.predicates:
            if arity == 0:
                self.predicates[key] = Bool(name)
            else:
                self.predicates[key] = Function(
                    name, *[BoolSort()] * arity, BoolSort())
        return self.predicates[key]

    def implies(self, items):
        return Implies(items[0], items[1])

    def iff(self, items):
        return items[0] == items[1]

    def xor(self, items):
        return Xor(items[0], items[1])

    def or_op(self, items):
        return Or(items[0], items[1])

    def and_op(self, items):
        return And(items[0], items[1])

    def not_op(self, items):
        return Not(items[0])

    def forall(self, items):
        var_name = str(items[0])
        var = Const(var_name, BoolSort())
        self.variables[var_name] = var
        return ForAll([var], items[1])

    def exists(self, items):
        var_name = str(items[0])
        var = Const(var_name, BoolSort())
        self.variables[var_name] = var
        return Exists([var], items[1])

    def predicate_with_args(self, items):
        name = str(items[0])
        args = items[1:]
        predicate = self.get_predicate(name, len(args))
        return predicate(*args)

    def predicate_without_args(self, items):
        name = str(items[0])
        return self.get_predicate(name)

    def term_variable(self, items):
        return self.get_var(str(items[0]))

    def term_constant(self, items):
        return Const(str(items[0]), BoolSort())

    def equals_num(self, items):
        var_name = str(items[0])
        num_val = float(items[1])

        pred_name = f"{var_name}_equals_{num_val}"
        if pred_name not in self.predicates:
            self.predicates[pred_name] = Bool(pred_name)

        self.comparison_variables[var_name] = True

        return self.predicates[pred_name]

    def less_than(self, items):
        var_name = str(items[0])
        num_val = float(items[1])

        pred_name = f"{var_name}_less_than_{num_val}"
        if pred_name not in self.predicates:
            self.predicates[pred_name] = Bool(pred_name)

        self.comparison_variables[var_name] = True

        return self.predicates[pred_name]

    def greater_than(self, items):
        var_name = str(items[0])
        num_val = float(items[1])

        pred_name = f"{var_name}_greater_than_{num_val}"
        if pred_name not in self.predicates:
            self.predicates[pred_name] = Bool(pred_name)

        self.comparison_variables[var_name] = True

        return self.predicates[pred_name]

    def less_equal(self, items):
        var_name = str(items[0])
        num_val = float(items[1])

        pred_name = f"{var_name}_less_equal_{num_val}"
        if pred_name not in self.predicates:
            self.predicates[pred_name] = Bool(pred_name)

        self.comparison_variables[var_name] = True

        return self.predicates[pred_name]

    def greater_equal(self, items):
        var_name = str(items[0])
        num_val = float(items[1])

        pred_name = f"{var_name}_greater_equal_{num_val}"
        if pred_name not in self.predicates:
            self.predicates[pred_name] = Bool(pred_name)

        self.comparison_variables[var_name] = True

        return self.predicates[pred_name]

    def not_equal(self, items):
        var_name = str(items[0])
        num_val = float(items[1])

        pred_name = f"{var_name}_not_equal_{num_val}"
        if pred_name not in self.predicates:
            self.predicates[pred_name] = Bool(pred_name)

        self.comparison_variables[var_name] = True

        return self.predicates[pred_name]

    def equals_str(self, items):
        var_name = str(items[0])

        string_value = str(items[1])
        if string_value.startswith('"') and string_value.endswith('"'):
            string_value = string_value[1:-1]
        elif string_value.startswith("'") and string_value.endswith("'"):
            string_value = string_value[1:-1]

        pred_name = f"{var_name}_equals_str_{string_value}"
        if pred_name not in self.predicates:
            self.predicates[pred_name] = Bool(pred_name)

        self.comparison_variables[var_name] = True

        return self.predicates[pred_name]

    def not_equal_str(self, items):
        var_name = str(items[0])

        # Extract the string value by removing quotes
        string_value = str(items[1])
        if string_value.startswith('"') and string_value.endswith('"'):
            string_value = string_value[1:-1]
        elif string_value.startswith("'") and string_value.endswith("'"):
            string_value = string_value[1:-1]

        # Create a special predicate to represent this comparison
        pred_name = f"{var_name}_not_equal_str_{string_value}"
        if pred_name not in self.predicates:
            self.predicates[pred_name] = Bool(pred_name)

        # Mark this variable as used in comparison
        self.comparison_variables[var_name] = True

        return self.predicates[pred_name]


# === FOL Checker ===
class EnhancedFOLChecker:
    def __init__(self, parser):
        self.parser = parser
        self.timeout = 500000

    def check_entailment(self, conclusion: str) -> Tuple[bool, Optional[z3.ModelRef], str]:
        premises_check = self.parser.check_sat()
        if premises_check == unsat:
            return True, None, ""

        try:
            is_entailed, model, explanation = self._check_via_negation(
                conclusion)

            if is_entailed is not None:
                return is_entailed, model, explanation

            is_entailed, model, explanation = self._check_via_implication(
                conclusion)

            return is_entailed, model, explanation
        except z3.Z3Exception as e:
            return False, None, f""
        except Exception as e:
            return False, None, f""

    def _check_via_negation(self, conclusion: str) -> Tuple[Optional[bool], Optional[z3.ModelRef], str]:
        temp_solver = Solver()

        temp_solver.set("timeout", self.timeout)

        for assertion in self.parser.solver.assertions():
            temp_solver.add(assertion)

        conclusion_formula = self.parser.parse(conclusion)

        # Add negation of conclusion
        temp_solver.add(Not(conclusion_formula))

        result = temp_solver.check()

        if result == unsat:
            # If unsatisfiable, then premises entail conclusion
            return True, None, ""
        elif result == sat:
            # If satisfiable, then premises don't entail conclusion
            model = temp_solver.model()
            counterexample = self._generate_counterexample(
                model, conclusion_formula)
            return False, model, f""
        else:  # unknown
            return None, None, ""

    def _check_via_implication(self, conclusion: str) -> Tuple[bool, Optional[z3.ModelRef], str]:
        temp_solver = Solver()
        temp_solver.set("timeout", self.timeout)
        conclusion_formula = self.parser.parse(conclusion)
        premises = []
        for assertion in self.parser.solver.assertions():
            premises.append(assertion)
        if not premises:
            temp_solver.add(Not(conclusion_formula))
            result = temp_solver.check()
            if result == unsat:
                return True, None, ""
            else:
                model = temp_solver.model() if result == sat else None
                return False, model, ""

        # Create the implication: (p1 ∧ p2 ∧ ... ∧ pn) → conclusion
        implication = Implies(And(premises), conclusion_formula)

        # Check if the implication is valid (its negation is unsatisfiable)
        temp_solver.add(Not(implication))
        result = temp_solver.check()

        if result == unsat:
            return True, None, ""
        elif result == sat:
            model = temp_solver.model()
            counterexample = self._generate_counterexample(
                model, conclusion_formula)
            return False, model, f""
        else:
            return False, None, ""

    def _generate_counterexample(self, model: z3.ModelRef, conclusion_formula) -> str:
        return ""

    def check_logical_equivalence(self, formula1: str, formula2: str) -> Tuple[bool, str]:
        try:
            f1 = self.parser.parse(formula1)
            f2 = self.parser.parse(formula2)
        except Exception as e:
            return False, f""

        # Check if f1 → f2 and f2 → f1
        s1 = Solver()
        s1.add(Not(Implies(f1, f2)))

        s2 = Solver()
        s2.add(Not(Implies(f2, f1)))

        result1 = s1.check()
        result2 = s2.check()

        if result1 == unsat and result2 == unsat:
            return True, ""

        explanation = []
        if result1 == sat:
            model = s1.model()
            counter = self._generate_counterexample(model, f2)
            explanation.append(
                f"")

        if result2 == sat:
            model = s2.model()
            counter = self._generate_counterexample(model, f1)
            explanation.append(
                f"")

        return False, "; ".join(explanation)

    def check_validity(self, formula: str) -> Tuple[bool, Optional[z3.ModelRef], str]:
        try:
            parsed_formula = self.parser.parse(formula)
        except Exception as e:
            return False, None, f""

        solver = Solver()
        solver.add(Not(parsed_formula))
        result = solver.check()

        if result == unsat:
            return True, None, ""
        elif result == sat:
            model = solver.model()
            counterexample = self._generate_counterexample(
                model, parsed_formula)
            return False, model, f""
        else:
            return False, None, ""

    def generate_all_models(self, max_models: int = 10) -> List[Tuple[z3.ModelRef, str]]:
        models = []

        solver = self.parser.solver.translate()

        for i in range(max_models):
            if solver.check() != sat:
                break

            model = solver.model()
            explanation = self._explain_model(model)
            models.append((model, explanation))

            block = []
            for d in model:
                const = d()
                if is_bool(const):
                    if is_true(model[d]):
                        block.append(const != True)
                    else:
                        block.append(const != False)

            solver.add(Or(block))

        return models

    def _explain_model(self, model: z3.ModelRef) -> str:
        explanation_parts = []

        variables = self.parser.transformer.variables
        predicates = self.parser.transformer.predicates

        # Process and format comparison predicates in a readable way
        comparison_predicates = []
        normal_predicates = []

        for pred_key, pred in predicates.items():
            if pred in model and is_true(model[pred]):
                name = pred_key

                # Try to parse comparison predicates
                if "_equals_" in name:
                    parts = name.split("_equals_")
                    var = parts[0]
                    val = parts[1]
                    comparison_predicates.append(f"{var} = {val}")
                elif "_less_than_" in name:
                    parts = name.split("_less_than_")
                    var = parts[0]
                    val = parts[1]
                    comparison_predicates.append(f"{var} < {val}")
                elif "_greater_than_" in name:
                    parts = name.split("_greater_than_")
                    var = parts[0]
                    val = parts[1]
                    comparison_predicates.append(f"{var} > {val}")
                elif "_less_equal_" in name:
                    parts = name.split("_less_equal_")
                    var = parts[0]
                    val = parts[1]
                    comparison_predicates.append(f"{var} ≤ {val}")
                elif "_greater_equal_" in name:
                    parts = name.split("_greater_equal_")
                    var = parts[0]
                    val = parts[1]
                    comparison_predicates.append(f"{var} ≥ {val}")
                elif "_not_equal_" in name:
                    parts = name.split("_not_equal_")
                    var = parts[0]
                    val = parts[1]
                    comparison_predicates.append(f"{var} ≠ {val}")
                else:
                    normal_predicates.append(name)

        # Add comparison predicates first
        if comparison_predicates:
            explanation_parts.extend(comparison_predicates)

        # Add boolean variables - excluding those used in comparisons
        for var_name, var in variables.items():
            if var_name in self.parser.transformer.comparison_variables:
                continue

            if var in model:
                value = model[var]
                if is_true(value):
                    explanation_parts.append(f"{var_name} = True")

        # Add normal predicates
        for pred_name in normal_predicates:
            explanation_parts.append(f"{pred_name} = True")

        return ", ".join(explanation_parts) if explanation_parts else "Empty model"

    def get_unsat_core(self, conclusion: str) -> Tuple[bool, List[int], str]:
        is_entailed, _, explanation = self.check_entailment(conclusion)
        if not is_entailed:
            return False, [], ""

        s = Solver()
        s.set("unsat_core", True)
        s.set("timeout", self.timeout)

        premise_strings = []
        tracked_premises = []

        for i, assertion in enumerate(self.parser.solver.assertions()):
            p_name = f"p{i}"
            premise_strings.append(p_name)
            p = Bool(p_name)
            s.assert_and_track(Implies(p, assertion), p_name)
            tracked_premises.append(p)

        # Add the negation of the conclusion
        conclusion_formula = self.parser.parse(conclusion)
        s.add(Not(conclusion_formula))

        # Check satisfiability
        result = s.check(tracked_premises)

        if result == unsat:
            core = s.unsat_core()
            core_indices = [int(str(c)[1:]) for c in core]
            core_premises = [str(self.parser.solver.assertions()[i])
                             for i in core_indices]

            return True, core_indices, f""
        else:
            return False, [], ""
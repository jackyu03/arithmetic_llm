import random


class ExpressionGenerator:
    def __init__(self, min_depth=1, max_depth=2, num_range=(1, 20), invalid_rate=0.1):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_range = num_range
        self.invalid_rate = invalid_rate

    

    def generate(self, current_depth=0, return_depth=False, target_depth=None):
        # Choose target depth for the entire expression at the root
        if current_depth == 0 and target_depth is None:
            target_depth = random.randint(self.min_depth, self.max_depth)
            
        # Fallback if somehow not set
        if target_depth is None:
            target_depth = self.max_depth

        # At target_depth, we MUST generate a number (leaf)
        if current_depth >= target_depth:
            res = str(random.randint(self.num_range[0], self.num_range[1]))
            return (res, 0) if return_depth else res
            
        # Below target_depth, we must expand at least one path to reach target_depth.
        # But we don't want every branch to be a straight line to target_depth.
        # 30% chance to be a leaf node early, EXCEPT if doing so makes it impossible 
        # for ANY branch to reach target_depth.
        # Since we are expanding an operation here, one of the two branches MUST 
        # be allowed to reach the target depth.
        
        op = random.choice(['+', '-'])
        
        # Decide which branch is the 'deep' branch that will definitely reach target_depth
        deep_branch_is_left = random.choice([True, False])
        
        # Generate Left Branch
        if current_depth < target_depth - 1 and not deep_branch_is_left and random.random() < 0.3:
            # We can safely terminate early because the right branch will go deep
            left_res = str(random.randint(self.num_range[0], self.num_range[1]))
            left_d = 0
            left = left_res
        else:
            if return_depth:
                left, left_d = self.generate(current_depth + 1, return_depth=True, target_depth=target_depth)
            else:
                left = self.generate(current_depth + 1, target_depth=target_depth)
                left_d = 0 # Dummy value

        # Generate Right Branch
        if current_depth < target_depth - 1 and deep_branch_is_left and random.random() < 0.3:
            # We can safely terminate early because the left branch already went deep
            right_res = str(random.randint(self.num_range[0], self.num_range[1]))
            right_d = 0
            right = right_res
        else:
            if return_depth:
                right, right_d = self.generate(current_depth + 1, return_depth=True, target_depth=target_depth)
            else:
                right = self.generate(current_depth + 1, target_depth=target_depth)
                right_d = 0 # Dummy value

        max_d = max(left_d, right_d) + 1

        if random.random() < self.invalid_rate:    
            error_type = random.choice(['missing_operand_right', 
                                        'missing_operand_left',
                                        'extra_operator++', 
                                        'extra_operator--',
                                        'unbalanced_paren_right',
                                        'unbalanced_paren_left',
                                        'arbitrary'
                                        ])
            if error_type == 'missing_operand_right':
                res = f"{left} +"
            elif error_type == 'missing_operand_left':
                res = f"+ {right}"
            elif error_type == 'extra_operator++':
                res = f"{left} ++ {right}"
            elif error_type == 'extra_operator--':
                res = f"{left} -- {right}"
            elif error_type == 'unbalanced_paren_right':
                res = f"({left} {op} {right}"
            elif error_type == 'unbalanced_paren_left':
                res = f"{left} {op} {right})"
            else:  # arbitrary error
                res = self._generate_invalid()
            return (res, max_d) if return_depth else res
        else:
            # Randomly decide to wrap in parentheses for visual structure
            if current_depth > 0:
                res = f"({left} {op} {right})"
            else:
                res = f"{left} {op} {right}"
            return (res, max_d) if return_depth else res
    
    def _generate_invalid(self):
        # Keep numeric tokens within num_range, even for invalid expressions.
        tokens = ['+', '-', '(', ')']
        length = random.randint(2, 20)
        parts = []
        for _ in range(length):
            if random.random() < 0.4:
                parts.append(str(random.randint(*self.num_range)))
            else:
                parts.append(random.choice(tokens))
        # Avoid accidental digit concatenation across numeric tokens.
        out = []
        for i, tok in enumerate(parts):
            out.append(tok)
            if i < len(parts) - 1:
                next_tok = parts[i + 1]
                if tok.isdigit() and next_tok.isdigit():
                    out.append(' ')
                elif random.random() < 0.3:
                    out.append(' ')
        return ''.join(out)

if __name__ == "__main__":
    # Usage
    for _ in range(5):
        generator = ExpressionGenerator(min_depth=1, max_depth=5, invalid_rate=0.1)
        new_expr = generator.generate()

        print(f"Generated Expression: {new_expr}")

    for _ in range(5):
        generator = ExpressionGenerator(min_depth=1, max_depth=5, invalid_rate=-1.0)
        new_expr = generator.generate()

        print(f"Generated Expression: {new_expr}")

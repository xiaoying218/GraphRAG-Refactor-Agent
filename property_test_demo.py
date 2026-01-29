import pytest
from hypothesis import given, settings, strategies as st

# ==========================================
# 1. 待测代码 (System Under Test)
# ==========================================
def ai_generated_sort(data):
    """
    这是一个由 AI 生成的有缺陷的排序算法。
    它不仅排序，还悄悄利用 set() 去重了。
    """
    if not data:
        return []
    # BUG: 这里使用了 set，导致如果输入中有重复元素，输出长度会变短！
    return sorted(list(set(data)))

# ==========================================
# 2. 传统单元测试 (Example-based Testing)
# ==========================================
class TestTraditional:
    """
    传统的测试方法：依赖开发者手动编写具体的输入输出。
    """
    
    def test_basic_sort(self):
        # 这个测试能通过，让开发者误以为代码是好的
        input_data = [3, 1, 2]
        expected = [1, 2, 3]
        assert ai_generated_sort(input_data) == expected

    def test_negative_numbers(self):
        # 这个也能通过
        input_data = [-1, -3, -2]
        expected = [-3, -2, -1]
        assert ai_generated_sort(input_data) == expected

    # 缺陷：除非开发者"极其聪明"地想到了测试重复元素，
    # 否则这个隐蔽的 Bug 就会溜进生产环境。

# ==========================================
# 3. 属性测试 (Property-based Testing)
# ==========================================
class TestPropertyBased:
    """
    属性测试：我们不写具体的例子，而是定义"真理"（Invariants）。
    让 Hypothesis 自动轰炸我们的代码。
    """

    # @given 装饰器会让 hypothesis 自动生成测试数据
    # st.lists(st.integers()) 表示：生成包含任意整数的列表
    @given(input_list=st.lists(st.integers()))
    @settings(max_examples=500) # 告诉它尝试生成 500 种不同的随机输入
    def test_sort_invariants(self, input_list):
        # 运行代码
        result = ai_generated_sort(input_list)

        # --- 定义核心属性 (Properties) ---

        # 属性 1: 有序性
        # 结果必须是升序排列的
        for i in range(len(result) - 1):
            assert result[i] <= result[i+1]

        # 属性 2: 元素守恒 (Conservation of Elements)
        # 输入里有的元素，输出里必须也要有（这能发现丢数据的问题）
        for x in input_list:
            assert x in result

        # 属性 3: 长度不变性 (Length Invariant)
        # 排序不应该改变列表的长度！
        # 【注意】这一行断言将会直接击溃上面的 Bug 代码
        assert len(result) == len(input_list)
from inspect import getdoc, signature


def add_num(a: int, b: int) -> int:
    """Add two Num"""

    return a + b


def main() -> None:
    for _, para in dict(signature(add_num).parameters).items():
        print(f"Parameter: {para.name}, Type: {para.annotation}")

    print(getdoc(add_num))


if __name__ == "__main__":
    main()

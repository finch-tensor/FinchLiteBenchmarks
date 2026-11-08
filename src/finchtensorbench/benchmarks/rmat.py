def choose_seed(rng, idxs, lg_dims):
    S = np.ones(())
    for _ in idxs:
        s = rng.choice([-1, 1]) * rng.random()
        S = np.multiply.outer(S, [0.5 - s / 2, 0.5 + s / 2])
    lg_size = sum(lg_dims[idx.name] for idx in idxs)
    return S * 2 ** -((1 - skewed_fractions(rng, SKEW)) * lg_size)

def choose(rng, options):
    return options[rng.integers(len(options))]


def shuffled(rng, items):
    return tuple(items[p] for p in rng.permutation(len(items)))


def skewed_fractions(rng, draws, size=None):
    """
    Fractions in [0, 1] distributed as the largest of draws uniform draws,
    which skews toward 1 the more draws there are: toward larger programs and
    denser inputs.
    """
    return rng.random(size) ** (1 / draws)


def block_seeds(S, lg_shape) -> list[np.ndarray]:
    """
    The seed of each block of a tensor of shape 2 ** lg_shape, with the skew of
    the seed S and a share of its log density for each bit of the block, so
    the density of the tensor is that of S.
    """
    density = np.sum(S)
    lg_size = sum(lg_shape)
    return [
        S / density * density ** (math.log2(math.prod(shape)) / lg_size)
        for shape in block_shapes(lg_shape, BLOCK_BITS)
    ]


def input_record(S, lg_shape, rng) -> dict[str, Any]:
    """
    The seeds of an input of shape 2 ** lg_shape with the seed S, and the
    version of numpy which draws its blocks from them and their checksum, which
    record_inputs checks when it draws them again.
    """
    seeds = block_seeds(S, lg_shape)
    rng_seed = int(rng.integers(2**63))
    tensor = kronecker_rmat(
        lg_shape, seeds, np.random.default_rng(rng_seed), BLOCK_BITS
    )
    return {
        "lg_shape": lg_shape,
        "rmat_seeds": [b.tolist() for b in seeds],
        "rng_seed": rng_seed,
        "numpy": np.__version__,
        "checksum": tensor.checksum(),
    }


def seed_lg_densities(prgm, input_fields, lg_dims, seeds):
    """
    The log nonzeros of the inputs whose blocks block_seeds splits from seeds,
    and the log expected products of the output of prgm, each as a fraction of
    its log size. By the mixed product property, the expected products are the
    product over levels of the sum of prgm run on the seeds of the level, which
    are the same at levels where the same fields have bits left.
    """
    query, _ = prgm.bodies
    output = query.lhs.tns
    fields = {**input_fields, output: query.lhs.idxs}
    lg_size = {
        tns: sum(lg_dims[idx.name] for idx in idxs) for tns, idxs in fields.items()
    }
    nnzs = {tns: round(np.sum(seeds[tns]) * 2 ** lg_size[tns]) for tns in input_fields}
    products = math.prod(nnzs.values())
    done = 0
    for lg in sorted(set(lg_dims.values())):
        level = {}
        for tns, idxs in input_fields.items():
            S = np.asarray(seeds[tns]) / np.sum(seeds[tns])
            out = tuple(d for d, idx in enumerate(idxs) if lg_dims[idx.name] < lg)
            level[tns] = fl.asarray(S.sum(axis=out, keepdims=True))
        (prod,) = LogicInterpreter()(prgm, level)
        products *= np.sum(prod.to_numpy()) ** (lg - done)
        done = lg

    def fraction(nnz, lg):
        return min(np.log2(max(nnz, 1)) / lg, 1.0) if lg else 1.0

    return [
        *(fraction(nnzs[tns], lg_size[tns]) for tns in input_fields),
        fraction(products, lg_size[output]),
    ]

def lg_size(size: float) -> int:
    n = int(size)
    if n <= 0 or n & (n - 1) or n != size:
        raise ValueError(f"Expected a power of two, got {size}")
    return n.bit_length() - 1


def block_shapes(lg_shape: Sequence[int], block_bits: int) -> list[tuple[int, ...]]:
    """
    The shape of each block of a tensor of shape 2 ** lg_shape, in which a
    dimension without a t-th digit has extent 1.
    """
    n_blocks = max((-(-lg // block_bits) for lg in lg_shape), default=0)
    return [
        tuple(2 ** min(block_bits, max(lg - t * block_bits, 0)) for lg in lg_shape)
        for t in range(n_blocks)
    ]


def rmat_block(S, lg_block: Sequence[int], rng) -> np.ndarray:
    """
    A boolean array of shape 2 ** lg_block whose S.sum() * 2 ** sum(lg_block)
    true cells are the first distinct cells which balls dropped by R-MAT with
    the seed S land in.
    """
    S = np.asarray(S, dtype=float)
    shape = tuple(2**lg for lg in lg_block)
    size = math.prod(shape)
    nnz = round(S.sum() * size)
    block = np.zeros(size, dtype=bool)
    if nnz:
        # At each level a ball picks an entry of S, and each axis with bits
        # left takes that entry's bit along it.
        P = np.ones(())
        bits = []
        for level in range(max(lg_block, default=0)):
            done = tuple(d for d, lg in enumerate(lg_block) if lg <= level)
            P = np.multiply.outer(P, S.sum(axis=done) / S.sum())
            bits += [(d, level) for d, lg in enumerate(lg_block) if level < lg]
        # The bits of an axis, from the top, make its coordinate.
        order = sorted(range(len(bits)), key=bits.__getitem__)
        P = P.transpose(order).reshape(shape)
        # The order in which balls first land in cells is the order of
        # exponential draws over the probabilities of the cells.
        with np.errstate(divide="ignore"):
            keys = rng.exponential(size=size) / P.ravel()
        block[np.argpartition(keys, nnz - 1)[:nnz]] = True
    return block.reshape(shape)
Concepts
========

-  Gereneralized approach to data aggregation and decomposition

-  all variables must have same dimensions
-  variables can be either `extensive or
   intensive <https://en.wikipedia.org/wiki/Intensive_and_extensive_properties>`__
-  all dimensions can have (multiple) hierarchies
-  hierarchies all start at a single scalar root
-  example for alternative hierarchies for a time dimension

.. mermaid::

    graph TD;
        * --- Jan;
        * --- Feb;
        * --- *...[...];
        * --- Dec;
        Jan --- 01-01;
        Jan --- 01-02;
        Jan --- 01...[...];
        Jan --- 01-31;
        01-31 --- 00:00;
        01-31 --- 01:00;
        01-31 --- 01-31...[...];
        01-31 --- 23:00;

.. mermaid::

    graph TD;
        * --- 1;
        * --- 2;
        * --- *...[...];
        * --- 8760;

-  Disaggregation maps data from one level in a hierarchy to the next
   one further down, aggregation to the next one up conversely, using
   level specific factors, which are usually all 1.0 or sum up to 1.0
   for each item

Examples
--------

Disaggregation (extensive) from A to B
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example: Distribute population by area (factors are shares of sub-areas
to total area)

.. mermaid::

    graph TD;
        * --- A1;
        * --- A2;
        A1 -- 0.5 --> B1;
        A1 -- 0.5 --> B2;
        A2 -- 0.4 --> B3;
        A2 -- 0.4 --> B4;
        A2 -- 0.2 --> B5;

Aggregation (extensive) from B to A
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example: Aggregate population (simply add up)

.. mermaid::

    flowchart TD;
        B1 -- 1 --> A1;
        B2 -- 1 --> A1;
        B3 -- 1 --> A2;
        B4 -- 1 --> A2;
        B5 -- 1 --> A2;
        A1 --- *
        A2 --- *

Disaggregation (intensive) from A to B
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example: Solar radiation

.. mermaid::

    graph TD;
        * --- A1;
        * --- A2;
        A1 -- 1 --> B1;
        A1 -- 1 --> B2;
        A2 -- 1 --> B3;
        A2 -- 1 --> B4;
        A2 -- 1 --> B5;

Aggregation (intensive) from B to A
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example: Solar radiation (weighted average by area)

.. mermaid::

    flowchart TD;
        B1 -- 0.5 --> A1;
        B2 -- 0.5 --> A1;
        B3 -- 0.4 --> A2;
        B4 -- 0.4 --> A2;
        B5 -- 0.2 --> A2;
        A1 --- *
        A2 --- *

Calculations
------------

Calculations can only be performed on variables when all dimensions are
at the same hierarchy level for all dimensions.

To achieve this, we disaggregate them all to the lowest level first.

At the end, we aggregate to the desired output level.

A change in level (aggregation or disaggregation) can be represented by
a matrix multiplication

.. math::

        D := \text{Data Matrix} \\
        T := \text{Transformation Matrix} \\
        D' = D \cdot T \\

Note: the data matrix will have more than one dimension. In the
examples, the first dimension represents all dimensions that are not
part of the transformation.

Disaggregation (extensive) from A(n=2) to B(n=5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

        D = \begin{pmatrix}
            0 & 1 \\
            2 & 3 \\
            4 & 5 \\
        \end{pmatrix} \\

        T = \begin{pmatrix}
            0.5 & 0.5 & \cdot & \cdot & \cdot \\
            \cdot & \cdot & 0.4 & 0.4 & 0.2
        \end{pmatrix}

        D' = \begin{pmatrix}
            0.0 & 0.0 & 0.4 & 0.4 & 0.2 \\
            1.0 & 1.0 & 1.2 & 1.2 & 0.6 \\
            2.0 & 2.0 & 2.0 & 2.0 & 1.0 \\
        \end{pmatrix}

Aggregation (extensive) from B(n=5) to A(n=2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

        D = \begin{pmatrix}
            0.0 & 0.0 & 0.4 & 0.4 & 0.2 \\
            1.0 & 1.0 & 1.2 & 1.2 & 0.6 \\
            2.0 & 2.0 & 2.0 & 2.0 & 1.0 \\
        \end{pmatrix}

        T = \begin{pmatrix}
            1 & \cdot \\
            1 & \cdot \\
            \cdot & 1 \\
            \cdot & 1 \\
            \cdot & 1
        \end{pmatrix}

        D' = \begin{pmatrix}
            0 & 1 \\
            2 & 3 \\
            4 & 5 \\
        \end{pmatrix} \\

Multiple transformations can be combined into a single transformation
Matrix.

.. math::

        T = T_1 \cdot T_2

Implementation with numpy
-------------------------

.. code:: python

    >>> .dot()
    array([[0. , 0. , 0.4, 0.4, 0.2],
           [1. , 1. , 1.2, 1.2, 0.6],
           [2. , 2. , 2. , 2. , 1. ]])
    >>>

    D = np.array([
        [0, 1],
        [2, 3],
        [4, 5]]
    )
    T = np.array([
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.4, 0.2]
    ])
    D2 = D.dot(T)

    # more generally, if there is more dimensions:

    D2 = np.matmul(D.swapaxes(i_dim, n_dims-1), T).swapaxes(n_dims-1, i_dim)

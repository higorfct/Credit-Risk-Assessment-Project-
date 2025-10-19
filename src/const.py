# const.py

sql_query = '''
SELECT 
    c.Profissao AS profession,
    c.TempoProfissao AS years_in_profession,
    c.Renda AS income,
    c.TipoResidencia AS residence_type,
    c.Escolaridade AS education,
    c.Score AS score,
    EXTRACT(YEAR FROM AGE(c.DataNascimento)) AS age,
    c.Dependentes AS dependents,
    c.EstadoCivil AS marital_status,
    pf.NomeComercial AS product,
    pc.ValorSolicitado AS requested_value,
    pc.ValorTotalBem AS total_asset_value,
    CASE 
        WHEN COUNT(p.Status) FILTER (WHERE p.Status = 'Vencido') > 0 THEN 'bad'
        ELSE 'good'
    END AS class
FROM clientes c
JOIN PedidoCredito pc ON c.ClienteID = pc.ClienteID
JOIN ProdutosFinanciados pf ON pc.ProdutoID = pf.ProdutoID
LEFT JOIN ParcelasCredito p ON pc.SolicitacaoID = p.SolicitacaoID
WHERE pc.Status = 'Aprovado'
GROUP BY c.ClienteID, pf.NomeComercial, pc.ValorSolicitado, pc.ValorTotalBem;
'''


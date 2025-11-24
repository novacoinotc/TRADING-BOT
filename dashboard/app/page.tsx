'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Dashboard() {
  const [data, setData] = useState<any>(null);
  const [closedTrades, setClosedTrades] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch status
        const statusResponse = await fetch(`${API_URL}/api/status`);
        if (!statusResponse.ok) throw new Error('API error');
        const statusData = await statusResponse.json();
        setData(statusData);

        // Fetch portfolio (historial de trades)
        try {
          const portfolioResponse = await fetch(`${API_URL}/portfolio`);
          if (portfolioResponse.ok) {
            const portfolioData = await portfolioResponse.json();
            // Obtener últimas 20 operaciones cerradas, más recientes primero
            const trades = (portfolioData.closed_trades || []).slice(-20).reverse();
            setClosedTrades(trades);
          }
        } catch (portfolioErr) {
          console.error('Error fetching portfolio:', portfolioErr);
        }

        setError(null);
        setLastUpdate(new Date());
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Función helper para formatear fecha
  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${day}/${month} ${hours}:${minutes}`;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-white text-xl">Cargando...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="bg-red-500/10 border border-red-500 rounded-lg p-6 max-w-md">
          <h2 className="text-red-500 text-xl font-bold mb-2">Error de Conexión</h2>
          <p className="text-gray-300">No se puede conectar con el bot.</p>
          <p className="text-gray-400 text-sm mt-2">{error}</p>
        </div>
      </div>
    );
  }

  const totalPnL = data?.positions?.total_unrealized_pnl || 0;
  const pnlColor = totalPnL >= 0 ? 'text-green-400' : 'text-red-400';

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-8">
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">
              🤖 Trading Bot Dashboard
            </h1>
            <p className="text-gray-400">v2.0 - Binance Futures Testnet</p>
          </div>
          <div className="text-right">
            <div className="flex items-center gap-2 justify-end">
              <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse"></div>
              <span className="text-green-400 font-semibold">Online</span>
            </div>
            {lastUpdate && (
              <p className="text-gray-500 text-sm mt-1">
                Última actualización: {lastUpdate.toLocaleTimeString()}
              </p>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              💰 Balance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div>
                <p className="text-gray-400 text-sm">Total</p>
                <p className="text-3xl font-bold text-white">
                  ${data?.balance?.total?.toLocaleString() || '0.00'}
                </p>
              </div>
              <div className="flex justify-between">
                <div>
                  <p className="text-gray-400 text-xs">Disponible</p>
                  <p className="text-green-400 font-semibold">
                    ${data?.balance?.available?.toLocaleString() || '0.00'}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400 text-xs">En Posiciones</p>
                  <p className="text-blue-400 font-semibold">
                    ${data?.balance?.in_positions?.toLocaleString() || '0.00'}
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              📊 Posiciones
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div>
                <p className="text-gray-400 text-sm">Abiertas</p>
                <p className="text-3xl font-bold text-white">
                  {data?.positions?.count || 0}
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-xs">P&L No Realizado</p>
                <p className={`text-2xl font-bold ${pnlColor}`}>
                  {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              🧠 RL Agent
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div>
                <p className="text-gray-400 text-sm">Total Trades</p>
                <p className="text-3xl font-bold text-white">
                  {data?.rl_agent?.total_trades || 0}
                </p>
              </div>
              <div className="flex justify-between">
                <div>
                  <p className="text-gray-400 text-xs">Exploration</p>
                  <p className="text-purple-400 font-semibold">
                    {((data?.rl_agent?.exploration_rate || 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-gray-400 text-xs">Q-Table</p>
                  <p className="text-purple-400 font-semibold">
                    {data?.rl_agent?.q_table_size || 0}
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {data?.positions?.list && data.positions.list.length > 0 && (
        <Card className="bg-slate-800/50 border-slate-700 max-w-7xl mx-auto mt-6">
          <CardHeader>
            <CardTitle className="text-white">Posiciones Abiertas</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left text-gray-400 font-semibold p-3">Symbol</th>
                    <th className="text-left text-gray-400 font-semibold p-3">Side</th>
                    <th className="text-right text-gray-400 font-semibold p-3">Entry</th>
                    <th className="text-right text-gray-400 font-semibold p-3">Qty</th>
                    <th className="text-right text-gray-400 font-semibold p-3">P&L</th>
                    <th className="text-center text-gray-400 font-semibold p-3">Leverage</th>
                  </tr>
                </thead>
                <tbody>
                  {data.positions.list.map((pos: any, idx: number) => (
                    <tr key={idx} className="border-b border-slate-700/50">
                      <td className="p-3">
                        <span className="text-white font-bold">{pos.symbol}</span>
                      </td>
                      <td className="p-3">
                        <Badge
                          className={
                            pos.side === 'LONG'
                              ? 'bg-green-500/20 text-green-400 border-green-500/50'
                              : 'bg-red-500/20 text-red-400 border-red-500/50'
                          }
                        >
                          {pos.side}
                        </Badge>
                      </td>
                      <td className="text-right text-white p-3">
                        ${pos.entry_price.toLocaleString()}
                      </td>
                      <td className="text-right text-gray-300 p-3">
                        {pos.quantity.toFixed(4)}
                      </td>
                      <td className="text-right p-3">
                        <span
                          className={`font-bold ${
                            pos.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}
                        >
                          {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                        </span>
                      </td>
                      <td className="text-center text-white p-3">{pos.leverage}x</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {(!data?.positions?.list || data.positions.list.length === 0) && (
        <Card className="bg-slate-800/50 border-slate-700 max-w-7xl mx-auto mt-6">
          <CardContent className="p-12 text-center">
            <p className="text-gray-400 text-lg">📭 Sin posiciones abiertas</p>
            <p className="text-gray-500 text-sm mt-2">
              El bot está monitoreando el mercado...
            </p>
          </CardContent>
        </Card>
      )}

      {/* HISTORIAL DE OPERACIONES */}
      <Card className="bg-slate-800/50 border-slate-700 max-w-7xl mx-auto mt-6">
        <CardHeader>
          <CardTitle className="text-white">📜 Historial de Operaciones</CardTitle>
        </CardHeader>
        <CardContent>
          {closedTrades.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-gray-400 text-sm border-b border-slate-700">
                    <th className="text-left py-3 px-4">#</th>
                    <th className="text-left py-3 px-4">Par</th>
                    <th className="text-left py-3 px-4">Tipo</th>
                    <th className="text-left py-3 px-4">Leverage</th>
                    <th className="text-right py-3 px-4">P&L</th>
                    <th className="text-left py-3 px-4">Fecha</th>
                  </tr>
                </thead>
                <tbody>
                  {closedTrades.map((trade, index) => (
                    <tr
                      key={trade.id || index}
                      className="border-b border-slate-700/50 hover:bg-slate-700/30 transition"
                    >
                      <td className="py-3 px-4 text-gray-300">{index + 1}</td>

                      <td className="py-3 px-4 text-white font-semibold">{trade.symbol}</td>

                      <td className="py-3 px-4">
                        <Badge
                          className={
                            trade.side === 'LONG'
                              ? 'bg-green-500/20 text-green-400 border-green-500/50'
                              : 'bg-red-500/20 text-red-400 border-red-500/50'
                          }
                        >
                          {trade.side}
                        </Badge>
                      </td>

                      <td className="py-3 px-4 text-gray-300">{trade.leverage}x</td>

                      <td className="text-right py-3 px-4">
                        <span
                          className={`font-bold ${
                            trade.pnl > 0 ? 'text-green-400' : 'text-red-400'
                          }`}
                        >
                          {trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                        </span>
                      </td>

                      <td className="py-3 px-4 text-gray-400 text-sm">
                        {formatDate(trade.timestamp)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              📭 Aún no hay operaciones cerradas
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

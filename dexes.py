"""Децентрализованные биржи: построение «виртуального стакана» из котировок агрегаторов.

У AMM нет книги заявок, поэтому глубина восстанавливается по кривой price impact:
агрегатор опрашивается на возрастающих размерах сделки, а разница между соседними
котировками даёт предельную цену и объём очередного «уровня» стакана.

Используются публичные API без ключей: KyberSwap, OpenOcean, ParaSwap, SushiSwap, LI.FI,
Jupiter (Solana), а также отдельные протоколы PancakeSwap и Uniswap — их котировки
получаются через маршрутизатор OpenOcean с фильтром по пулам конкретного протокола.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import permutations, product
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from urllib.parse import urlparse

TIMEOUT = 25
ZERO_ADDR = "0x0000000000000000000000000000000000000001"
HEADERS = {"User-Agent": "arb-calculator/1.0", "Accept": "application/json"}


# --------------------------------------------------------------------------------------
# Сети и токены
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    symbol: str
    address: str
    decimals: int


def _t(sym: str, addr: str, dec: int) -> Tuple[str, Token]:
    return sym, Token(sym, addr, dec)


CHAINS: Dict[str, dict] = {
    "BNB Chain": {
        "kyber": 'bsc', "openocean": 'bsc', "paraswap": 56,
        "lifi": 56, "sushi": 56, "jupiter": False,
        "pancake": 'BNB Chain', "uniswap": 'BNB Chain',
        "gas_symbol": "BNB",
        "tokens": dict([
            _t("USDT", "0x55d398326f99059ff775485246999027b3197955", 18),
            _t("USDC", "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d", 18),
            _t("FDUSD", "0xc5f0f7b66764f6ec8c8dff7ba683102295e16409", 18),
            _t("BUSD", "0xe9e7cea3dedca5984780bafc599bd69add087d56", 18),
            _t("DAI", "0x1af3f329e8be154074d8769d1ffa4ee058b1dbc3", 18),
            _t("TUSD", "0x40af3827f39d0eacbf4a168f8d4ee67c121d11c9", 18),
            _t("USD1", "0x8d0d000ee44948fc98c9b98a4fa4921476f08b0d", 18),
            _t("ETH", "0x2170ed0880ac9a755fd29b2688956bd959f933f8", 18),
            _t("WBNB", "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c", 18),
            _t("BNB", "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE", 18),
            _t("BTCB", "0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c", 18),
            _t("SolvBTC", "0x4aae823a6a0b376de6a78e74ecc5b079d38cbcf7", 18),
            _t("CAKE", "0x0e09fabb73bd3ade0a17ecc321fd13a19e81ce82", 18),
            _t("TWT", "0x4b0f1812e5df2a09796481ff14017e6005508003", 18),
            _t("XRP", "0x1d2f0da169ceb9fc7b3144628db156f3f6c60dbe", 18),
            _t("ADA", "0x3ee2200efb3400fabb9aacf31297cbdd1d435d47", 18),
            _t("DOGE", "0xba2ae424d960c26247dd6c32edc70b295c744c43", 8),
            _t("DOT", "0x7083609fce4d1d8dc0c979aab8c869ea2c873402", 18),
            _t("LINK", "0xf8a0bf9cf54bb92f17374d9e9a321e6a111a51bd", 18),
            _t("LTC", "0x4338665cbb7b2485a8855a139b75d5e34ab0db94", 18),
            _t("UNI", "0xbf5140a22578168fd562dccf235e5d43a02ce9b1", 18),
            _t("AAVE", "0xfb6115445bff7b52feb98650c87f44907e58f802", 18),
            _t("SUSHI", "0x947950bcc74888a40ffa2593c5798f11fc9124c4", 18),
            _t("PEPE", "0x25d887ce7a35172c62febfd67a1856f20faebb00", 18),
            _t("SHIB", "0x2859e4544C4bB03966803b044A93563Bd2D0DD4D", 18),
            _t("FLOKI", "0xfb5b838b6cfeedc2873ab27866079ac55363d37e", 9),
        ]),
    },
    "Ethereum": {
        "kyber": 'ethereum', "openocean": 'eth', "paraswap": 1,
        "lifi": 1, "sushi": 1, "jupiter": False,
        "pancake": 'Ethereum', "uniswap": 'Ethereum',
        "gas_symbol": "ETH",
        "tokens": dict([
            _t("USDT", "0xdac17f958d2ee523a2206206994597c13d831ec7", 6),
            _t("USDC", "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", 6),
            _t("FDUSD", "0xc5f0f7b66764f6ec8c8dff7ba683102295e16409", 18),
            _t("BUSD", "0x4fabb145d64652a948d72533023f6e7a623c7c53", 18),
            _t("DAI", "0x6b175474e89094c44da98b954eedeac495271d0f", 18),
            _t("TUSD", "0x0000000000085d4780b73119b644ae5ecd22b376", 18),
            _t("USDe", "0x4c9edd5852cd905f086c759e8383e09bff1e68b3", 18),
            _t("USD1", "0x8d0d000ee44948fc98c9b98a4fa4921476f08b0d", 18),
            _t("PYUSD", "0x6c3ea9036406852006290770bedfcaba0e23a0e8", 6),
            _t("crvUSD", "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e", 18),
            _t("WETH", "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", 18),
            _t("ETH", "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE", 18),
            _t("WBTC", "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599", 8),
            _t("cbBTC", "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf", 8),
            _t("SolvBTC", "0x7A56E1C57C7475CCf742a1832B028F0456652F97", 18),
            _t("LINK", "0x514910771af9ca656af840dff83e8264ecf986ca", 18),
            _t("UNI", "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984", 18),
            _t("AAVE", "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9", 18),
            _t("CRV", "0xd533a949740bb3306d119cc777fa900ba034cd52", 18),
            _t("SUSHI", "0x6b3595068778dd592e39a122f4f5a5cf09c90fe2", 18),
            _t("LDO", "0x5a98fcbea516cf06857215779fd812ca3bef1b32", 18),
            _t("MKR", "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2", 18),
            _t("ARB", "0xb50721bcf8d664c30412cfbc6cf7a15145234ad1", 18),
            _t("PEPE", "0x6982508145454ce325ddbe47a25d4ec3d2311933", 18),
            _t("SHIB", "0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce", 18),
            _t("FLOKI", "0xcf0c122c6b73ff809c693db761e7baebe62b6a2e", 9),
        ]),
    },
    "Arbitrum": {
        "kyber": 'arbitrum', "openocean": 'arbitrum', "paraswap": 42161,
        "lifi": 42161, "sushi": 42161, "jupiter": False,
        "pancake": 'Arbitrum', "uniswap": 'Arbitrum',
        "gas_symbol": "ETH",
        "tokens": dict([
            _t("USDC", "0xaf88d065e77c8cC2239327C5EDb3A432268e5831", 6),
            _t("DAI", "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1", 18),
            _t("TUSD", "0x4D15a3A2286D883AF0AA1B3f21367843FAc63E07", 18),
            _t("WETH", "0x82af49447d8a07e3bd95bd0d56f35241523fbab1", 18),
            _t("ETH", "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE", 18),
            _t("WBTC", "0x2f2a2543b76a4166549f7aab2e75bef0aefc5b0f", 8),
            _t("cbBTC", "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf", 8),
            _t("SolvBTC", "0x3647c54c4c2c65bc7a2d63c0da2809b399dbbdc0", 18),
            _t("LINK", "0xf97f4df75117a78c1a5a0dbb814af92458539fb4", 18),
            _t("UNI", "0xfa7f8980b0f1e64a2062791cc3b0871572f1f7f0", 18),
            _t("CRV", "0x11cdb42b0eb46d95f990bedd4695a6e3fa034978", 18),
            _t("SUSHI", "0xd4d42f0b6def4ce0383636770ef773390d85c61a", 18),
            _t("LDO", "0x13ad51ed4f1b7e9dc168d8a00cb3f4ddd85efa60", 18),
            _t("MKR", "0x2e9a6Df78E42a30712c10a9Dc4b1C8656f8F2879", 18),
            _t("ARB", "0x912ce59144191c1204e64559fe8253a0e49e6548", 18),
            _t("GNO", "0xa0b862f60edef4452f25b4160f177db44deb6cf1", 18),
            _t("STG", "0x6694340fc020c5E6B96567843da2df01b2CE1eb6", 18),
            _t("GRT", "0x23a941036ae778ac51ab04cea08ed6e2fe103614", 18),
            _t("SolvBTC.BBN", "0x346c574c56e1a4aaa8dc88cda8f7eb12b39947ab", 18),
            _t("ETHFI", "0x7189fb5b6504bbff6a852b13b7b82a3c118fdc27", 18),
            _t("PENDLE", "0x0c880f6761f1af8d9aa9c466984b80dab9a8c9e8", 18),
        ]),
    },
    "Base": {
        "kyber": 'base', "openocean": 'base', "paraswap": 8453,
        "lifi": 8453, "sushi": 8453, "jupiter": False,
        "pancake": 'Base', "uniswap": 'Base',
        "gas_symbol": "ETH",
        "tokens": dict([
            _t("USDT", "0xfde4c96c8593536e31f229ea8f37b2ada2699bb2", 6),
            _t("USDC", "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913", 6),
            _t("USDbC", "0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA", 6),
            _t("DAI", "0x50c5725949a6f0c72e6c4a641f24049a917db0cb", 18),
            _t("WETH", "0x4200000000000000000000000000000000000006", 18),
            _t("ETH", "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE", 18),
            _t("WBTC", "0x0555e30da8f98308edb960aa94c0db47230d2b9c", 8),
            _t("cbBTC", "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf", 8),
            _t("SolvBTC", "0x3B86Ad95859b6AB773f55f8d94B4b9d443EE931f", 18),
            _t("CAKE", "0x3055913c90fcc1a6ce9a358911721eeb942013a1", 18),
            _t("DOT", "0x8d010bf9c26881788b4e6bf5fd1bdc358c8f90b8", 18),
            _t("LINK", "0x88fb150bdc53a65fe94dea0c9ba0a6daf8c6e196", 18),
            _t("AAVE", "0x63706e401c06ac8513145b7687a14804d17f814b", 18),
            _t("CRV", "0x8ee73c484a26e0a5df2ee2a4960b789967dd0415", 18),
            _t("ENA", "0x58538e6a46e07434d7e7375bc268d3cb839c0133", 18),
            _t("AERO", "0x940181a94a35a4569e4529a3cdfb74e38fd98631", 18),
            _t("STG", "0xe3b53af74a4bf62ae5511055290838050bf764df", 18),
            _t("SolvBTC.BBN", "0xC26C9099BD3789107888c35bb41178079B282561", 18),
            _t("EIGEN", "0x2081ab0d9ec9e4303234ab26d86b20b3367946ee", 18),
            _t("ETHFI", "0x6c240dda6b5c336df09a4d011139beaaa1ea2aa2", 18),
            _t("PENDLE", "0xa99f6e6785da0f5d6fb42495fe424bce029eeb3e", 18),
            _t("W", "0xb0ffa8000886e57f86dd5264b9582b2ad87b2b91", 18),
            _t("VIRTUAL", "0x0b3e328455c4059EEb9e3f84b5543F74E24e7E1b", 18),
            _t("BRETT", "0x532f27101965dd16442e59d40670faf5ebb142e4", 18),
            _t("DEGEN", "0x4ed4e862860bed51a9570b96d89af5e1b0efefed", 18),
            _t("MOG", "0x2da56acb9ea78330f947bd57c54119debda7af71", 18),
        ]),
    },
    "Polygon": {
        "kyber": 'polygon', "openocean": 'polygon', "paraswap": 137,
        "lifi": 137, "sushi": 137, "jupiter": False,
        "pancake": None, "uniswap": 'Polygon',
        "gas_symbol": "POL",
        "tokens": dict([
            _t("USDT", "0xc2132D05D31c914a87C6611C10748AEb04B58e8F", 6),
            _t("USDC", "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359", 6),
            _t("USDC.e", "0x2791bca1f2de4661ed88a30c99a7a9449aa84174", 6),
            _t("BUSD", "0x9C9e5fD8bbc25984B178FdCE6117Defa39d2db39", 18),
            _t("DAI", "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063", 18),
            _t("WETH", "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619", 18),
            _t("WBTC", "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6", 8),
            _t("LINK", "0x53E0bca35eC356BD5ddDFebbD1Fc0fD03FaBad39", 18),
            _t("UNI", "0xb33EaAd8d922B1083446DC23f610c2567fB5180f", 18),
            _t("AAVE", "0xD6DF932A45C0f255f85145f286eA0b292B21C90B", 18),
            _t("CRV", "0x172370d5cd63279efa6d502dab29171933a610af", 18),
            _t("SUSHI", "0x0b3f868e0be5597d5db7feb59e1cadbb0fdda50a", 18),
            _t("STG", "0x2f6f07cdcf3588944bf4c42ac74ff24bf56e7590", 18),
            _t("SAND", "0xbbba073c31bf03b8acf7c28ef0738decf3695683", 18),
            _t("DEGEN", "0x8a2870fb69A90000D6439b7aDfB01d4bA383A415", 18),
        ]),
    },
    "Optimism": {
        "kyber": 'optimism', "openocean": 'optimism', "paraswap": 10,
        "lifi": 10, "sushi": 10, "jupiter": False,
        "pancake": None, "uniswap": 'Optimism',
        "gas_symbol": "ETH",
        "tokens": dict([
            _t("USDT", "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58", 6),
            _t("USDC", "0x0b2c639c533813f4aa9d7837caf62653d097ff85", 6),
            _t("USDC.e", "0x7f5c764cbc14f9669b88837ca1490cca17c31607", 6),
            _t("DAI", "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1", 18),
            _t("WETH", "0x4200000000000000000000000000000000000006", 18),
            _t("ETH", "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE", 18),
            _t("WBTC", "0x68f180fcce6836688e9084f035309e29bf0a2095", 8),
            _t("LINK", "0x350a791bfc2c21f9ed5d10980dad2e2638ffa7f6", 18),
            _t("LDO", "0xFdb794692724153d1488CcdBE0C56c252596735F", 18),
            _t("OP", "0x4200000000000000000000000000000000000042", 18),
            _t("VELO", "0x9560e827aF36c94D2Ac33a39bCE1Fe78631088Db", 18),
            _t("STG", "0x296f55f8fb28e498b858d0bcda06d955b2cb3f97", 18),
            _t("PENDLE", "0xbc7b1ff1c6989f006a1185318ed4e7b5796e66e1", 18),
        ]),
    },
    "Avalanche": {
        "kyber": 'avalanche', "openocean": 'avax', "paraswap": 43114,
        "lifi": 43114, "sushi": 43114, "jupiter": False,
        "pancake": None, "uniswap": 'Avalanche',
        "gas_symbol": "AVAX",
        "tokens": dict([
            _t("USDT", "0x9702230a8ea53601f5cd2dc00fdbc13d4df4a8c7", 6),
            _t("USDC", "0xb97ef9ef8734c71904d8002f8b6bc66dd9c48a6e", 6),
            _t("USDC.e", "0xA7D7079b0FEaD91F3e65f86E8915Cb59c1a4C664", 6),
            _t("BUSD", "0x9c9e5fd8bbc25984b178fdce6117defa39d2db39", 18),
            _t("WETH.e", "0x49D5c2BdFfac6CE2BFdB6640F4F80f226bc10bAB", 18),
            _t("BTC.b", "0x152b9d0FdC40C096757F570A51E494bd4b943E50", 8),
            _t("WBTC.e", "0x50b7545627a5162f82a992c33b87adc75187b218", 8),
            _t("SolvBTC", "0xbc78d84ba0c46dfe32cf2895a19939c86b81a777", 18),
            _t("WAVAX", "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7", 18),
            _t("AVAX", "0x0000000000000000000000000000000000000000", 18),
            _t("STG", "0x2F6F07CDcf3588944Bf4C42aC74ff24bF56e7590", 18),
            _t("SolvBTC.BBN", "0xcc0966d8418d412c599a6421b760a847eb169a8c", 18),
            _t("PENDLE", "0xfb98b335551a418cd0737375a2ea0ded62ea213b", 18),
        ]),
    },
    "Linea": {
        "kyber": 'linea', "openocean": 'linea', "paraswap": None,
        "lifi": 59144, "sushi": 59144, "jupiter": False,
        "pancake": None, "uniswap": None,
        "gas_symbol": "ETH",
        "tokens": dict([
            _t("USDT", "0xa219439258ca9da29e9cc4ce5596924745e12b93", 6),
            _t("USDC", "0x176211869ca2b568f2a7d4ee941e073a821ee1ff", 6),
            _t("BUSD", "0x7d43AABC515C356145049227CeE54B608342c0ad", 18),
            _t("DAI", "0x4af15ec2a0bd43db75dd04e62faa3b8ef36b00d5", 18),
            _t("WETH", "0xe5D7C2a44FfDDf6b295A15c148167daaAf5Cf34f", 18),
            _t("ETH", "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE", 18),
            _t("BNB", "0xf5C6825015280CdfD0b56903F9F8B5A2233476F5", 18),
            _t("WBTC", "0x3aAB2285ddcDdaD8edf438C1bAB47e1a9D05a9b4", 8),
            _t("AVAX", "0x5471ea8f739dd37E9B81Be9c5c77754D8AA953E4", 18),
            _t("MATIC", "0x265B25e22bcd7f10a5bD6E6410F10537Cc7567e8", 18),
            _t("weETH", "0x1Bf74C010E6320bab11e2e5A532b5AC15e0b8aA6", 18),
            _t("ezETH", "0x2416092f143378750bb29b79ed961ab195cceea5", 18),
        ]),
    },
    "zkSync Era": {
        "kyber": None, "openocean": 'zksync', "paraswap": None,
        "lifi": 324, "sushi": 324, "jupiter": False,
        "pancake": None, "uniswap": None,
        "gas_symbol": "ETH",
        "tokens": dict([
            _t("USDT", "0x493257fD37EDB34451f62EDf8D2a0C418852bA4C", 6),
            _t("USDC", "0x1d17cbcf0d6d143135ae902365d2e5e2a16538d4", 6),
            _t("USDC.e", "0x3355df6D4c9C3035724Fd0e3914dE96A5a83aaf4", 6),
            _t("BUSD", "0x2039bb4116B4EFc145Ec4f0e2eA75012D6C0f181", 18),
            _t("WETH", "0x5AEa5775959fBC2557Cc8789bC1bf90A239D9a91", 18),
            _t("ETH", "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE", 18),
            _t("WBTC", "0xBBeB516fb02a01611cBBE0453Fe3c580D7281011", 8),
        ]),
    },
    "Sonic": {
        "kyber": 'sonic', "openocean": 'sonic', "paraswap": None,
        "lifi": 146, "sushi": 146, "jupiter": False,
        "pancake": None, "uniswap": None,
        "gas_symbol": "S",
        "tokens": dict([
            _t("USDT", "0x6047828dc181963ba44974801FF68e538dA5eaF9", 6),
            _t("USDC", "0x29219dd400f2bf60e5a23d13be72b486d4038894", 6),
            _t("WETH", "0x50c42dEAcD8Fc9773493ED674b675bE577f2634b", 18),
            _t("S", "0x0000000000000000000000000000000000000000", 18),
            _t("PENDLE", "0xf1ef7d2d4c0c881cd634481e0586ed5d2871a74b", 18),
        ]),
    },
    "Cronos": {
        "kyber": None, "openocean": 'cronos', "paraswap": None,
        "lifi": 25, "sushi": 25, "jupiter": False,
        "pancake": None, "uniswap": None,
        "gas_symbol": "CRO",
        "tokens": dict([
            _t("USDT", "0x66e428c3f67a68878562e79a0234c1f83c208770", 6),
            _t("USDC", "0x3D7F2C478aAfdB65542BCB44bCeeC05849999d2D", 6),
            _t("USDC.e", "0xf951eC28187D9E5Ca673Da8FE6757E6f0Be5F77C", 6),
            _t("BUSD", "0x6ab6d61428fde76768d7b45d8bfeec19c6ef91a8", 18),
            _t("DAI", "0xf2001b145b43032aaf5ee2884e456ccd805f677d", 18),
            _t("TUSD", "0x87EFB3ec1576Dec8ED47e58B832bEdCd86eE186e", 18),
            _t("WETH", "0xe44fd7fcb2b1581822d0c862b68222998a0c299a", 18),
            _t("BNB", "0xfa9343c3897324496a05fc75abed6bac29f8a40f", 18),
            _t("WCRO", "0x5c7f8a570d578ed84e63fdfa7b1ee72deae1ae23", 18),
            _t("XRP", "0xb9Ce0dd29C91E02d4620F57a66700Fc5e41d6D15", 6),
            _t("DOGE", "0x1a8E39ae59e5556B56b76fCBA98d22c9ae557396", 8),
            _t("SHIB", "0xbED48612BC69fA1CaB67052b42a95FB30C1bcFee", 18),
            _t("SOL", "0xc9DE0F3e08162312528FF72559db82590b481800", 9),
            _t("AVAX", "0x765277EebeCA2e31912C9946eAe1021199B39C61", 18),
            _t("MATIC", "0xad79AC3c5a5c15C6B9194F5568e451b3fc3C2B40", 18),
            _t("CRO", "0x0000000000000000000000000000000000000000", 18),
            _t("FTM", "0xB44a9B6905aF7c801311e8F4E76932ee959c663C", 18),
            _t("ATOM", "0xB888d8Dd1733d72681b30c00ee76BDE93ae7aa93", 6),
        ]),
    },
    "Gnosis": {
        "kyber": None, "openocean": 'xdai', "paraswap": None,
        "lifi": 100, "sushi": 100, "jupiter": False,
        "pancake": None, "uniswap": None,
        "gas_symbol": "xDAI",
        "tokens": dict([
            _t("USDT", "0x4ECaBa5870353805a9F068101A40E0f32ed605C6", 6),
            _t("USDC", "0xDDAfbb505ad214D7b80b1f830fcCc89B60fb7A83", 6),
            _t("USDC.e", "0x2a22f9c3b484c3629090FeED35F17Ff8F88f76F0 ", 6),
            _t("WXDAI", "0xe91D153E0b41518A2Ce8Dd3D7944Fa863463a97d", 18),
            _t("WETH", "0x6A023CCd1ff6F2045C3309768eAd9E68F978f6e1", 18),
            _t("WBTC", "0x8e5bBbb09Ed1ebdE8674Cda39A0c169401db4252", 8),
            _t("xDAI", "0x0000000000000000000000000000000000000000", 18),
            _t("LINK", "0xE2e73A1c69ecF83F464EFCE6A5be353a37cA09b2", 18),
            _t("UNI", "0x4537e328Bf7e4eFA29D05CAeA260D7fE26af9D74", 18),
            _t("SUSHI", "0x2995D1317DcD4f0aB89f4AE60F3f020A4F17C7CE", 18),
            _t("wstETH", "0x6c76971f98945ae98dd7d4dfca8711ebea946ea6", 18),
            _t("sDAI", "0xaf204776c7245bF4147c2612BF6e5972Ee483701", 18),
            _t("GNO", "0x9c58bacc331c9aa871afd802db6379a98e80cedb", 18),
        ]),
    },
    "Solana": {
        "kyber": None, "openocean": 'solana', "paraswap": None,
        "lifi": None, "sushi": None, "jupiter": True,
        "pancake": None, "uniswap": None,
        "gas_symbol": "SOL",
        "tokens": dict([
            _t("USDT", "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", 6),
            _t("USDC", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 6),
            _t("PYUSD", "2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GXo", 6),
            _t("XRP", "Ga2AXHpfAF6mv2ekZwcsJFqu7wB4NV331qNH7fW9Nst8", 6),
            _t("LINK", "CWE8jPTUYhdCTZYWPTe1o5DFqfdjzWKc9WKz6rSjQUdG", 6),
            _t("UNI", "8FU95xFJhUUkyyCLU13HSzDLs7oC4QZdXQHL6SCeab36", 8),
            _t("SUSHI", "AR1Mtgh7zAtxuxGd2XPovXPVjcSdY3i4rQYisNadjfKy", 6),
            _t("BONK", "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263", 5),
            _t("JUP", "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN", 6),
            _t("RAY", "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R", 6),
            _t("SOL", "So11111111111111111111111111111111111111112", 9),
            _t("HNT", "HqB7uswoVg4suaQiDP3wjxob1G5WdZ144zhdStwMCq7e", 6),
            _t("TRUMP", "6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN", 6),
            _t("VIRTUAL", "3iQL8BFS2vE7mww4ehAqQHAsbmRNCrPxizWAT2Zfyr9y", 9),
            _t("WIF", "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm", 6),
            _t("JTO", "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL", 9),
            _t("PYTH", "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3", 6),
        ]),
    },
}

# Минимальный интервал между запросами к одному хосту (бесплатные лимиты агрегаторов)
_MIN_INTERVAL = {
    "open-api.openocean.finance": 1.15,
    "li.quest": 1.0,
    "api.paraswap.io": 0.35,
    "api.sushi.com": 0.25,
}
_LOCKS: Dict[str, threading.Lock] = {}
_LAST: Dict[str, float] = {}
_REGISTRY_LOCK = threading.Lock()


def _throttle(host: str) -> None:
    delay = _MIN_INTERVAL.get(host, 0.0)
    if not delay:
        return
    with _REGISTRY_LOCK:
        lock = _LOCKS.setdefault(host, threading.Lock())
    with lock:
        wait = delay - (time.monotonic() - _LAST.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _LAST[host] = time.monotonic()


def _get(url: str, params: dict, attempts: int = 3) -> dict:
    """GET с уважением к бесплатным лимитам: троттлинг по хосту и повтор при 429."""
    host = urlparse(url).netloc
    last: Optional[Exception] = None
    for i in range(attempts):
        _throttle(host)
        r = requests.get(url, params=params, timeout=TIMEOUT, headers=HEADERS)
        if r.status_code == 429:
            last = requests.HTTPError("429 Too Many Requests", response=r)
            time.sleep(1.5 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise ValueError("площадка ограничила частоту запросов, попробуйте через минуту") from last


# --------------------------------------------------------------------------------------
# Адаптеры агрегаторов: возвращают (полученное количество, комиссия газа в USD)
# --------------------------------------------------------------------------------------

def _kyber(chain: str, tin: Token, tout: Token, amount_in: int) -> Tuple[float, float]:
    slug = CHAINS[chain]["kyber"]
    d = _get(
        f"https://aggregator-api.kyberswap.com/{slug}/api/v1/routes",
        {"tokenIn": tin.address, "tokenOut": tout.address, "amountIn": str(amount_in)},
    )
    rs = d["data"]["routeSummary"]
    return int(rs["amountOut"]) / 10 ** tout.decimals, float(rs.get("gasUsd") or 0)


def _openocean(chain: str, tin: Token, tout: Token, amount_in: int,
               dex_ids: Optional[str] = None) -> Tuple[float, float]:
    slug = CHAINS[chain]["openocean"]
    human = amount_in / 10 ** tin.decimals
    params = {
        "inTokenAddress": tin.address,
        "outTokenAddress": tout.address,
        "amount": f"{human:.10f}".rstrip("0").rstrip("."),
        "gasPrice": 5,
    }
    if dex_ids:
        params["enabledDexIds"] = dex_ids
    d = _get(f"https://open-api.openocean.finance/v3/{slug}/quote", params)
    if d.get("code") != 200:
        raise ValueError(str(d.get("error") or d.get("message") or "ошибка OpenOcean"))
    t = d["data"]
    return int(t["outAmount"]) / 10 ** tout.decimals, 0.0


def _paraswap(chain: str, tin: Token, tout: Token, amount_in: int) -> Tuple[float, float]:
    net = CHAINS[chain]["paraswap"]
    d = _get(
        "https://api.paraswap.io/prices",
        {
            "srcToken": tin.address,
            "destToken": tout.address,
            "srcDecimals": tin.decimals,
            "destDecimals": tout.decimals,
            "amount": str(amount_in),
            "side": "SELL",
            "network": net,
        },
    )
    pr = d["priceRoute"]
    return int(pr["destAmount"]) / 10 ** tout.decimals, float(pr.get("gasCostUSD") or 0)


def _jupiter(chain: str, tin: Token, tout: Token, amount_in: int) -> Tuple[float, float]:
    d = _get(
        "https://lite-api.jup.ag/swap/v1/quote",
        {
            "inputMint": tin.address,
            "outputMint": tout.address,
            "amount": str(amount_in),
            "slippageBps": 50,
        },
    )
    return int(d["outAmount"]) / 10 ** tout.decimals, 0.0



def _sushi(chain: str, tin: Token, tout: Token, amount_in: int) -> Tuple[float, float]:
    cid = CHAINS[chain]["sushi"]
    d = _get(
        f"https://api.sushi.com/swap/v7/{cid}",
        {
            "tokenIn": tin.address,
            "tokenOut": tout.address,
            "amount": str(amount_in),
            "maxSlippage": 0.005,
            "sender": ZERO_ADDR,
        },
    )
    if d.get("status") != "Success" or not d.get("assumedAmountOut"):
        raise ValueError("SushiSwap не нашёл маршрут")
    return int(d["assumedAmountOut"]) / 10 ** tout.decimals, 0.0


def _lifi(chain: str, tin: Token, tout: Token, amount_in: int) -> Tuple[float, float]:
    cid = CHAINS[chain]["lifi"]
    try:
        d = _get(
            "https://li.quest/v1/quote",
            {
                "fromChain": cid, "toChain": cid,
                "fromToken": tin.address, "toToken": tout.address,
                "fromAmount": str(amount_in), "fromAddress": ZERO_ADDR,
                "slippage": 0.005, "skipSimulation": "true",
            },
        )
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", 0)
        if code == 429:
            raise ValueError("LI.FI: превышен лимит бесплатных запросов, попробуйте позже") from None
        raise ValueError(f"LI.FI: маршрут не найден (HTTP {code})") from None
    est = d.get("estimate") or {}
    out = est.get("toAmount")
    if not out:
        raise ValueError("LI.FI не нашёл маршрут")
    gas = sum(float(g.get("amountUSD") or 0) for g in (d.get("gasCosts") or []))
    return int(out) / 10 ** tout.decimals, gas


# --- Прямые котировки из блокчейна (без ключей, через публичные RPC) -----------
RPC_URLS: Dict[str, str] = {
    "BNB Chain": "https://bsc-rpc.publicnode.com",
    "Ethereum": "https://ethereum-rpc.publicnode.com",
    "Arbitrum": "https://arbitrum-one-rpc.publicnode.com",
    "Base": "https://base-rpc.publicnode.com",
    "Polygon": "https://polygon-bor-rpc.publicnode.com",
    "Optimism": "https://optimism-rpc.publicnode.com",
    "Avalanche": "https://avalanche-c-chain-rpc.publicnode.com",
}

# сеть -> (роутер V2, квотер V3, набор комиссий V3, hub-токен для маршрута через 2 пула)
ONCHAIN_VENUES: Dict[str, Dict[str, tuple]] = {
    "PancakeSwap": {
        "BNB Chain": ("0x10ED43C718714eb63d5aA57B78B54704E256024E", "0xB048Bbc1Ee6b733FFfCFb9e9CeF7375518e25997", (100, 500, 2500, 10000), "WBNB"),
        "Ethereum": ("0xEfF92A263d31888d860bD50809A8D171709b7b1c", "0xB048Bbc1Ee6b733FFfCFb9e9CeF7375518e25997", (100, 500, 2500, 10000), "WETH"),
        "Arbitrum": ("0x8cFe327CEc66d1C090Dd72bd0FF11d690C33a2Eb", "0xB048Bbc1Ee6b733FFfCFb9e9CeF7375518e25997", (100, 500, 2500, 10000), "WETH"),
        "Base": ("0x8cFe327CEc66d1C090Dd72bd0FF11d690C33a2Eb", "0xB048Bbc1Ee6b733FFfCFb9e9CeF7375518e25997", (100, 500, 2500, 10000), "WETH"),
    },
    "Uniswap": {
        "Ethereum": ("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", "0x61fFE014bA17989E743c5F6cB21bF9697530B21e", (100, 500, 3000, 10000), "WETH"),
        "Arbitrum": ("0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24", "0x61fFE014bA17989E743c5F6cB21bF9697530B21e", (100, 500, 3000, 10000), "WETH"),
        "Base": ("0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24", "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a", (100, 500, 3000, 10000), "WETH"),
        "Polygon": ("0xedf6066a2b290C185783862C7F4776A2C8077AD1", "0x61fFE014bA17989E743c5F6cB21bF9697530B21e", (100, 500, 3000, 10000), "WETH"),
        "Optimism": ("0x4A7b5Da61326A6379179b40d00F57E5bbDC962c2", "0x61fFE014bA17989E743c5F6cB21bF9697530B21e", (100, 500, 3000, 10000), "WETH"),
        "BNB Chain": ("0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24", "0x78D78E420Da98ad378D7799bE8f4AF69033EB077", (100, 500, 3000, 10000), "WBNB"),
        "Avalanche": ("0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24", "0xbe0F5544EC67e9B3b2D979aaA43f18Fd87E6257F", (100, 500, 3000, 10000), "WAVAX"),
    },
}

_GAS_UNITS = 180_000


def _w(x: int) -> str:
    return f"{x:064x}"


def _addr_word(a: str) -> str:
    return _w(int(a, 16))


def _v2_calldata(amount: int, path: List[str]) -> str:
    """getAmountsOut(uint256,address[])"""
    return "0x" + "d06ca61f" + _w(amount) + _w(64) + _w(len(path)) + "".join(_addr_word(p) for p in path)


def _v3_calldata(tin: str, tout: str, amount: int, fee: int) -> str:
    """quoteExactInputSingle((address,address,uint256,uint24,uint160))"""
    return "0xc6a5026a" + _addr_word(tin) + _addr_word(tout) + _w(amount) + _w(fee) + _w(0)


def _rpc_batch(chain: str, calls: List[Tuple[str, str]], with_gas_price: bool = False) -> List[Optional[str]]:
    url = RPC_URLS[chain]
    payload = [{"jsonrpc": "2.0", "id": i, "method": "eth_call",
                "params": [{"to": to, "data": data}, "latest"]}
               for i, (to, data) in enumerate(calls)]
    if with_gas_price:
        payload.append({"jsonrpc": "2.0", "id": len(calls), "method": "eth_gasPrice", "params": []})
    r = requests.post(url, json=payload, timeout=TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        raise ValueError("RPC-узел вернул ошибку")
    by_id = {item.get("id"): item.get("result") for item in data}
    return [by_id.get(i) for i in range(len(calls) + (1 if with_gas_price else 0))]


def _wrapped_address(chain: str, token: Token) -> str:
    """Нативная монета в роутерах представлена обёрнутым токеном."""
    if not token.address.lower().startswith("0xeeeeeeee"):
        return token.address
    for sym in (f"W{token.symbol}", "WETH", "WBNB"):
        t = CHAINS[chain]["tokens"].get(sym)
        if t and not t.address.lower().startswith("0xeeeeeeee"):
            return t.address
    raise ValueError(f"нет обёрнутого токена для {token.symbol}")


_NATIVE_USD: Dict[str, Tuple[float, float]] = {}


def _native_usd(chain: str, venue: str) -> float:
    hit = _NATIVE_USD.get(chain)
    if hit and time.monotonic() - hit[1] < 120:
        return hit[0]
    tokens = CHAINS[chain]["tokens"]
    hub_sym = ONCHAIN_VENUES[venue][chain][3]
    stable = next((tokens[s] for s in ("USDT", "USDC", "DAI", "USDC.e") if s in tokens), None)
    if stable is None or hub_sym not in tokens:
        return 0.0
    try:
        out = _onchain_amount_out(chain, venue, tokens[hub_sym], stable, 10 ** tokens[hub_sym].decimals)
        price = out / 10 ** stable.decimals
    except Exception:
        price = 0.0
    _NATIVE_USD[chain] = (price, time.monotonic())
    return price


def _onchain_amount_out(chain: str, venue: str, tin: Token, tout: Token, amount_in: int,
                        want_gas: bool = False) -> float:
    cfg = ONCHAIN_VENUES.get(venue, {}).get(chain)
    if not cfg:
        raise ValueError(f"{venue} не поддерживается в сети {chain}")
    v2_router, quoter, fees, hub_sym = cfg
    a_in = _wrapped_address(chain, tin)
    a_out = _wrapped_address(chain, tout)
    hub_tok = CHAINS[chain]["tokens"].get(hub_sym)
    hub = _wrapped_address(chain, hub_tok) if hub_tok else None

    calls: List[Tuple[str, str]] = [(v2_router, _v2_calldata(amount_in, [a_in, a_out]))]
    if hub and hub.lower() not in (a_in.lower(), a_out.lower()):
        calls.append((v2_router, _v2_calldata(amount_in, [a_in, hub, a_out])))
    n_v2 = len(calls)
    for f in fees:
        calls.append((quoter, _v3_calldata(a_in, a_out, amount_in, f)))

    res = _rpc_batch(chain, calls, with_gas_price=want_gas)
    gas_hex = res[len(calls)] if want_gas else None
    best = 0
    for i, raw in enumerate(res[:len(calls)]):
        if not raw or raw == "0x":
            continue
        try:
            if i < n_v2:
                h = raw[2:]
                cnt = int(h[64:128], 16)
                val = int(h[128 + 64 * (cnt - 1): 128 + 64 * cnt], 16)
            else:
                val = int(raw[2:66], 16)
        except Exception:
            continue
        best = max(best, val)
    if best <= 0:
        raise ValueError(f"{venue}: нет пула для {tin.symbol}/{tout.symbol} в сети {chain}")
    if want_gas:
        return best, gas_hex
    return best


def _onchain_venue(venue: str):
    def fn(chain: str, tin: Token, tout: Token, amount_in: int) -> Tuple[float, float]:
        best, gas_hex = _onchain_amount_out(chain, venue, tin, tout, amount_in, want_gas=True)
        gas_usd = 0.0
        try:
            if gas_hex:
                native = _native_usd(chain, venue)
                gas_usd = int(gas_hex, 16) * _GAS_UNITS / 1e18 * native
        except Exception:
            gas_usd = 0.0
        return best / 10 ** tout.decimals, gas_usd
    return fn


@dataclass
class Aggregator:
    name: str
    key: str
    fn: Callable[[str, Token, Token, int], Tuple[float, float]]

    def supports(self, chain: str) -> bool:
        return bool(CHAINS[chain].get(self.key))


AGGREGATORS: Dict[str, Aggregator] = {
    "KyberSwap": Aggregator("KyberSwap", "kyber", _kyber),
    "OpenOcean": Aggregator("OpenOcean", "openocean", _openocean),
    "ParaSwap": Aggregator("ParaSwap", "paraswap", _paraswap),
    "SushiSwap": Aggregator("SushiSwap", "sushi", _sushi),
    "LI.FI": Aggregator("LI.FI", "lifi", _lifi),
    "Jupiter": Aggregator("Jupiter", "jupiter", _jupiter),
    "PancakeSwap": Aggregator("PancakeSwap", "pancake", _onchain_venue("PancakeSwap")),
    "Uniswap": Aggregator("Uniswap", "uniswap", _onchain_venue("Uniswap")),
}


def aggregators_for(chain: str) -> List[str]:
    return [n for n, a in AGGREGATORS.items() if a.supports(chain)]


# --------------------------------------------------------------------------------------
# Виртуальный стакан
# --------------------------------------------------------------------------------------

@dataclass
class DexBooks:
    aggregator: str
    chain: str
    base: str
    quote: str
    ask: Optional[pd.DataFrame]
    bid: Optional[pd.DataFrame]
    gas_usd: float = 0.0
    spot: Optional[float] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.ask is not None and self.bid is not None


def _ladder(prices: List[float], sizes: List[float], side: str) -> pd.DataFrame:
    rows = [(p, s) for p, s in zip(prices, sizes) if s > 0 and np.isfinite(p) and p > 0]
    if not rows:
        raise ValueError("не удалось построить кривую цены")
    # монотонность: ask не должен убывать, bid — расти
    fixed = []
    for i, (p, s) in enumerate(rows):
        if i:
            prev = fixed[-1][0]
            p = max(p, prev) if side == "ask" else min(p, prev)
        fixed.append((p, s))
    df = pd.DataFrame(fixed, columns=["price", "size"])
    df.insert(0, "level", np.arange(1, len(df) + 1))
    return df


def fetch_dex_books(
    aggregator: str,
    chain: str,
    base_sym: str,
    quote_sym: str,
    probe_depth: float,
    steps: int = 8,
) -> DexBooks:
    """Строит ask/bid «стакан» глубиной probe_depth базового токена.

    probe_depth — сколько базового токена прогоняется через кривую (например, 20 WETH).
    steps       — число уровней; на каждый уровень уходит по одному запросу на сторону.
    """
    agg = AGGREGATORS[aggregator]
    cfg = CHAINS[chain]
    if not agg.supports(chain):
        return DexBooks(aggregator, chain, base_sym, quote_sym, None, None,
                        error=f"{aggregator} не поддерживает сеть {chain}")
    try:
        tb: Token = cfg["tokens"][base_sym]
        tq: Token = cfg["tokens"][quote_sym]
    except KeyError as e:
        return DexBooks(aggregator, chain, base_sym, quote_sym, None, None,
                        error=f"токен {e} не найден в сети {chain}")

    def sell_base(amount_base: float):
        return agg.fn(chain, tb, tq, int(round(amount_base * 10 ** tb.decimals)))

    def buy_base(amount_quote: float):
        return agg.fn(chain, tq, tb, int(round(amount_quote * 10 ** tq.decimals)))

    try:
        probe = max(probe_depth / (steps * 20), 10 ** -tb.decimals)
        spot_out, gas = sell_base(probe)
        spot = spot_out / probe
        if spot <= 0:
            raise ValueError("нулевая котировка")

        base_grid = np.linspace(probe_depth / steps, probe_depth, steps)
        quote_grid = base_grid * spot

        with ThreadPoolExecutor(max_workers=3) as pool:
            sells = list(pool.map(lambda b: sell_base(float(b))[0], base_grid))
        with ThreadPoolExecutor(max_workers=3) as pool:
            buys = list(pool.map(lambda q: buy_base(float(q))[0], quote_grid))

        # bid: продаём базу, получаем котировку
        bid_prices, bid_sizes, prev_q, prev_b = [], [], 0.0, 0.0
        for b, q in zip(base_grid, sells):
            db, dq = b - prev_b, q - prev_q
            if db > 0:
                bid_prices.append(dq / db)
                bid_sizes.append(db)
            prev_b, prev_q = b, q

        # ask: платим котировкой, получаем базу
        ask_prices, ask_sizes, prev_q, prev_b = [], [], 0.0, 0.0
        for q, b in zip(quote_grid, buys):
            db, dq = b - prev_b, q - prev_q
            if db > 0:
                ask_prices.append(dq / db)
                ask_sizes.append(db)
            prev_b, prev_q = b, q

        return DexBooks(
            aggregator, chain, base_sym, quote_sym,
            _ladder(ask_prices, ask_sizes, "ask"),
            _ladder(bid_prices, bid_sizes, "bid"),
            gas_usd=gas,
            spot=spot,
        )
    except Exception as exc:
        return DexBooks(aggregator, chain, base_sym, quote_sym, None, None,
                        error=str(exc)[:180])


def dex_spot(aggregator: str, chain: str, base_sym: str, quote_sym: str,
             amount: float = 1.0) -> Optional[float]:
    """Быстрая цена одного базового токена в котируемом."""
    agg = AGGREGATORS[aggregator]
    cfg = CHAINS[chain]
    try:
        tb, tq = cfg["tokens"][base_sym], cfg["tokens"][quote_sym]
        out, _ = agg.fn(chain, tb, tq, int(round(amount * 10 ** tb.decimals)))
        return out / amount
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Межплощадочный треугольный арбитраж
# --------------------------------------------------------------------------------------

@dataclass
class DexQuote:
    venue: str
    chain: str
    source: str
    target: str
    amount_in: float
    amount_out: Optional[float]
    gas_usd: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.amount_out is not None and self.amount_out > 0


def dex_quote(venue: str, chain: str, source: str, target: str, amount_in: float) -> DexQuote:
    """Возвращает котировку точного объёма свопа source → target."""
    try:
        aggregator = AGGREGATORS[venue]
        token_in = CHAINS[chain]["tokens"][source]
        token_out = CHAINS[chain]["tokens"][target]
        amount_out, gas_usd = aggregator.fn(
            chain, token_in, token_out, int(round(amount_in * 10 ** token_in.decimals))
        )
        if amount_out <= 0:
            raise ValueError("нулевая котировка")
        return DexQuote(venue, chain, source, target, amount_in, amount_out, gas_usd)
    except Exception as exc:
        return DexQuote(venue, chain, source, target, amount_in, None, error=str(exc)[:180])


USD_LIKE = {"USDT", "USDC", "DAI", "BUSD", "FDUSD", "TUSD", "USD1", "USDE", "PYUSD"}


def _cross_venue_triangle(
    chain: str,
    route: Tuple[str, str, str, str],
    venues: Tuple[str, str, str],
    start_amount: float,
) -> Optional[dict]:
    amount = start_amount
    gas_usd = 0.0
    legs = []
    for index, venue in enumerate(venues):
        quote = dex_quote(venue, chain, route[index], route[index + 1], amount)
        if not quote.ok:
            return None
        amount = quote.amount_out
        gas_usd += quote.gas_usd
        legs.append(quote)

    start = route[0]
    final_after_gas = amount - gas_usd if start.upper() in USD_LIKE else amount
    net_bps = (final_after_gas / start_amount - 1.0) * 10_000.0
    return {
        "Сеть": chain,
        "Маршрут": " → ".join(route),
        "Площадка A": venues[0],
        "Нога A": f"{route[0]} → {route[1]}",
        "Котировка A": f"{legs[0].amount_in:.8g} {route[0]} → {legs[0].amount_out:.8g} {route[1]}",
        "Площадка B": venues[1],
        "Нога B": f"{route[1]} → {route[2]}",
        "Котировка B": f"{legs[1].amount_in:.8g} {route[1]} → {legs[1].amount_out:.8g} {route[2]}",
        "Площадка C": venues[2],
        "Нога C": f"{route[2]} → {route[3]}",
        "Котировка C": f"{legs[2].amount_in:.8g} {route[2]} → {legs[2].amount_out:.8g} {route[3]}",
        f"Старт, {start}": start_amount,
        f"Финиш до газа, {start}": amount,
        "Газ, USD": gas_usd,
        f"Финиш после газа, {start}": final_after_gas,
        f"Прибыль, {start}": final_after_gas - start_amount,
        "Чистый спред, б.п.": net_bps,
    }


def cross_venue_triangles(
    chain: str,
    venues: List[str],
    assets: List[str],
    start_asset: str,
    start_amount: float,
    top: int = 50,
) -> Tuple[pd.DataFrame, dict]:
    """Перебирает треугольники, где минимум две ноги находятся на разных DEX.

    Все ноги выполняются в одной выбранной сети; мосты между сетями не
    используются. Для долларовых стартовых активов оценка газа вычитается из
    финального результата.
    """
    start = start_asset.upper()
    selected_assets = list(dict.fromkeys(asset.upper() for asset in assets if asset.strip()))
    selected_assets = [asset for asset in selected_assets if asset in CHAINS[chain]["tokens"]]
    selected_venues = [venue for venue in venues if venue in AGGREGATORS and AGGREGATORS[venue].supports(chain)]
    routes = [(start, first, second, start) for first, second in permutations([asset for asset in selected_assets if asset != start], 2)]
    venue_sets = [sequence for sequence in product(selected_venues, repeat=3) if len(set(sequence)) >= 2]
    jobs = [(route, venue_set) for route in routes for venue_set in venue_sets]
    rows = []
    failed = 0
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(jobs)))) as pool:
        futures = {
            pool.submit(_cross_venue_triangle, chain, route, venue_set, start_amount): (route, venue_set)
            for route, venue_set in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                failed += 1
            else:
                rows.append(result)

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("Чистый спред, б.п.", ascending=False).head(top).reset_index(drop=True)
    return table, {"routes": len(routes), "venue_sets": len(venue_sets), "completed": len(rows), "failed": failed}

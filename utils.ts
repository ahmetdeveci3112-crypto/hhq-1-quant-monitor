
export const formatPrice = (price: number | undefined | null): string => {
    if (price === undefined || price === null) return '0.00';

    // Convert to number just in case string is passed
    const numPrice = Number(price);
    if (isNaN(numPrice)) return '0.00';

    if (numPrice === 0) return '0.00';

    // Very small numbers (e.g. SHIB, PEPE)
    if (numPrice < 0.01) {
        return numPrice.toFixed(8); // Show 8 decimals for very small coins
    }

    // Small numbers (e.g. DOGE)
    if (numPrice < 1) {
        return numPrice.toFixed(6);
    }

    // Medium numbers (e.g. XRP, ADA)
    if (numPrice < 10) {
        return numPrice.toFixed(4);
    }

    // Large numbers (e.g. BTC, ETH)
    // Use toLocaleString for comma separation (e.g. 98,500.50)
    return numPrice.toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
};

export const formatCurrency = (amount: number | undefined | null): string => {
    if (amount === undefined || amount === null) return '$0.00';
    const num = Number(amount);
    return num.toLocaleString('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
};
